import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Iterator, Callable

from ghapi.core import GhApi
from fastcore.net import HTTP403ForbiddenError, HTTP404NotFoundError, HTTP422UnprocessableEntityError

PR_KEYWORDS = {
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
}

def parse_json_or_jsonl(path: str):
    """解析 JSON 或 JSONL 文件"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试作为 JSONL 解析
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

def parse_instance_id(instance_id: str, repo_name_override: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """从 instance_id 中提取仓库名和 PR 编号"""
    repo_name = repo_name_override
    pr_number = None
    if repo_name is None:
        # 尝试从不同格式的 instance_id 解析
        if "__" in instance_id and "-" in instance_id:
            left, right = instance_id.rsplit("-", 1)
            if "__" in left and right.isdigit():
                owner, repo = left.split("__", 1)
                repo_name = f"{owner}/{repo}"
                pr_number = int(right)
        elif "/" in instance_id and "_" in instance_id:
            left, right = instance_id.rsplit("_", 1)
            if right.isdigit():
                repo_name = left
                pr_number = int(right)
        elif "_" in instance_id:
            left, right = instance_id.rsplit("_", 1)
            if right.isdigit():
                repo_name = left.replace("__", "/")
                pr_number = int(right)
    
    # 尝试直接正则匹配末尾数字作为 PR 编号
    if pr_number is None:
        match = re.search(r"(\d+)$", instance_id)
        if match:
            pr_number = int(match.group(1))
            
    if repo_name is None or pr_number is None:
        return None, None
    return repo_name, pr_number

def call_api(
    api: GhApi,
    func: Callable,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    suppress_422: bool = False,
    **kwargs,) -> Optional[dict]:
    """
    API call wrapper with rate limit handling (checks every 5 minutes if rate limit is reset)

    Args:
        func (callable): API function to call
        **kwargs: keyword arguments to pass to API function
    Return:
        values (dict): response object of `func`
    """
    if owner is not None and "owner" not in kwargs:
        kwargs["owner"] = owner
    if repo is not None and "repo" not in kwargs:
        kwargs["repo"] = repo
    while True:
        try:
            values = func(**kwargs)
            return values
        except HTTP403ForbiddenError:
            while True:
                rl = api.rate_limit.get()
                owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
                token_prefix = token[:10] if isinstance(token, str) else "unknown"
                print(
                    f"[{owner_repo}] Rate limit exceeded for token {token_prefix}, "
                    f"waiting for 5 minutes, remaining calls: {rl.resources.core.remaining}"
                )
                if rl.resources.core.remaining > 0:
                    break
                time.sleep(60 * 5)
        except HTTP404NotFoundError:
            owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
            print(f"[{owner_repo}] Resource not found {kwargs}")
            return None
        except HTTP422UnprocessableEntityError:
            if not suppress_422:
                owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
                print(f"[{owner_repo}] Unprocessable request {kwargs}")
            return None

def extract_resolved_issues(api: GhApi, owner: str, repo: str, pull: dict) -> list[str]:
    """
    Extract list of issues referenced by a PR

    Args:
        pull (dict): PR dictionary object from GitHub
    Return:
        resolved_issues (list): list of issue numbers referenced by PR
    """
    # Define 1. issue number regex pattern 2. comment regex pattern 3. keywords
    issues_pat = re.compile(r"(\w+)\s+\#(\d+)")
    comments_pat = re.compile(r"(?s)<!--.*?-->")

    # Construct text to search over for issue numbers from PR body and commit messages
    title = getattr(pull, "title", None) if not isinstance(pull, dict) else pull.get("title")
    body = getattr(pull, "body", None) if not isinstance(pull, dict) else pull.get("body")
    number = getattr(pull, "number", None) if not isinstance(pull, dict) else pull.get("number")
    text = title if title else ""
    text += "\n" + (body if body else "")
    commits = get_all_loop(
        api,
        api.pulls.list_commits,
        owner=owner,
        repo=repo,
        pull_number=number,
        quiet=True,
    )
    commit_messages = [commit.commit.message for commit in commits]
    commit_text = "\n".join(commit_messages) if commit_messages else ""
    text += "\n" + commit_text
    # Remove comments from text
    text = comments_pat.sub("", text)
    # Look for issue numbers in text via scraping <keyword, number> patterns
    # print(f'for pr {number}, text: {text}')
    references = issues_pat.findall(text)
    resolved_issues_set = set()
    if references:
        for word, issue_num in references:
            if word.lower() in PR_KEYWORDS:
                resolved_issues_set.add(issue_num)
    return list(resolved_issues_set)

def get_all_loop(
    api: GhApi,
    func: Callable,
    per_page: int = 100,
    num_pages: Optional[int] = None,
    quiet: bool = False,
    **kwargs,) -> Iterator:
    """
    Return all values from a paginated API endpoint.

    Args:
        func (callable): API function to call
        per_page (int): number of values to return per page
        num_pages (int): number of pages to return
        quiet (bool): whether to print progress
        **kwargs: keyword arguments to pass to API function
    """
    page = 1
    args = {"per_page": per_page, **kwargs}
    while True:
        try:
            # Get values from API call
            values = func(**args, page=page)
            yield from values
            if len(values) == 0:
                break
            if not quiet:
                rl = api.rate_limit.get()
                owner = kwargs.get("owner")
                repo = kwargs.get("repo")
                owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
                print(
                    f"[{owner_repo}] Processed page {page} ({per_page} values per page). "
                    f"Remaining calls: {rl.resources.core.remaining}"
                )
            if num_pages is not None and page >= num_pages:
                break
            page += 1
        except Exception as e:
            # Rate limit handling
            owner = kwargs.get("owner")
            repo = kwargs.get("repo")
            owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
            print(f"[{owner_repo}] Error processing page {page} - {e}")
            while True:
                rl = api.rate_limit.get()
                if rl.resources.core.remaining > 0:
                    break
                print(
                    f"[{owner_repo}] Waiting for rate limit reset "
                    f"checking again in 5 minutes"
                )
                time.sleep(60 * 5)
    if not quiet:
        owner = kwargs.get("owner")
        repo = kwargs.get("repo")
        owner_repo = f"{owner}/{repo}" if owner and repo else "unknown/unknown"
        print(f"[{owner_repo}] Processed {(page - 1) * per_page + len(values)} values")

def _fetch_timeline_events(token: str, owner: str, repo: str, issue_number: int, per_page: int = 100) -> list:
    """
    直接通过 requests 调用 GitHub Timeline API，绕过 ghapi。
    GET /repos/{owner}/{repo}/issues/{issue_number}/timeline
    返回所有分页合并后的事件列表（原始 dict）。
    """
    import requests as _requests

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_events = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline"
        resp = _requests.get(url, headers=headers, params={"per_page": per_page, "page": page})
        if resp.status_code == 404 or resp.status_code == 410:
            print(f"[{owner}/{repo}] Timeline API {resp.status_code} for #{issue_number}")
            break
        if resp.status_code == 403:
            # rate limit
            print(f"[{owner}/{repo}] Timeline API rate limited, waiting 5 min...")
            time.sleep(60 * 5)
            continue
        if resp.status_code != 200:
            print(f"[{owner}/{repo}] Timeline API unexpected status {resp.status_code} for #{issue_number}")
            break

        events = resp.json()
        if not isinstance(events, list) or len(events) == 0:
            break
        all_events.extend(events)
        if len(events) < per_page:
            break
        page += 1

    return all_events


def get_mentioned_issues_and_prs(
    api: GhApi,
    owner: str,
    repo: str,
    number: int,
    target_type: str = "pr",
    per_page: int = 100,
    num_pages: Optional[int] = None,):
    """
    通过 GitHub Timeline API 的 cross-referenced 事件，查找 mention 到指定 PR/Issue 的其他 PR/Issue。

    target_type="pr":  返回该 PR merge 之后 mention 到它的 closed issue 和 merged PR（本仓库）
    target_type="issue": 返回所有 mention 到该 issue 的 merged PR（本仓库，不限时间）
    """
    owner = owner.strip()
    repo = repo.strip()
    number = int(number)

    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def parse_github_time(ts):
        if not isinstance(ts, str) or not ts:
            return None
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    # 获取目标 PR 的 merged_at（仅 target_type="pr" 时需要）
    target_merged_at = None
    if target_type == "pr":
        target_pr = call_api(api, api.pulls.get, owner=owner, repo=repo, pull_number=number)
        if target_pr is None:
            return {"merged_prs": [], "closed_issues": []}
        target_merged_at = parse_github_time(_get(target_pr, "merged_at"))
        if target_merged_at is None:
            return {"merged_prs": [], "closed_issues": []}

    # 获取 token
    token = getattr(api, "token", None)

    # 直接用 requests 获取 timeline 事件
    all_events = _fetch_timeline_events(token, owner, repo, number, per_page=per_page)
    print(f"[{owner}/{repo}#{number}] Timeline: {len(all_events)} events, "
          f"cross-referenced: {sum(1 for e in all_events if isinstance(e, dict) and e.get('event') == 'cross-referenced')}")

    merged_prs = []
    closed_issues = []
    seen_pr_numbers = set()
    seen_issue_numbers = set()

    for event in all_events:
        if not isinstance(event, dict):
            continue
        if event.get("event") != "cross-referenced":
            continue

        # PR 目标：只保留 merge 之后的 mention
        if target_type == "pr":
            event_created_at = parse_github_time(event.get("created_at"))
            if event_created_at is None or event_created_at <= target_merged_at:
                continue
        # issue 目标：不做时间限制

        source = event.get("source")
        if not isinstance(source, dict):
            continue
        source_issue = source.get("issue")
        if not isinstance(source_issue, dict):
            continue

        src_number = source_issue.get("number")
        if src_number is None:
            continue

        # 检查是否本仓库
        src_repo_url = source_issue.get("repository_url", "")
        src_html_url = source_issue.get("html_url", "")
        is_same_repo = (
            src_repo_url.rstrip("/").endswith(f"/{owner}/{repo}")
            or f"/{owner}/{repo}/" in src_html_url
        )
        if not is_same_repo:
            continue

        is_pr = source_issue.get("pull_request") is not None

        if is_pr:
            # 引用源是 PR：只保留本仓库的 merged PR
            if src_number in seen_pr_numbers:
                continue
            if target_type == "pr" and src_number == number:
                continue

            pr_detail = call_api(api, api.pulls.get, owner=owner, repo=repo, pull_number=src_number)
            if pr_detail is None:
                continue
            if not _get(pr_detail, "merged_at"):
                continue

            seen_pr_numbers.add(src_number)
            merged_prs.append({
                "number": src_number,
                "title": _get(pr_detail, "title"),
                "state": _get(pr_detail, "state"),
                "html_url": _get(pr_detail, "html_url"),
            })
        else:
            # 引用源是 issue：仅 PR 目标时收集 closed issue
            if target_type != "pr":
                continue
            if src_number in seen_issue_numbers:
                continue
            if source_issue.get("state") != "closed":
                continue

            seen_issue_numbers.add(src_number)
            closed_issues.append({
                "number": src_number,
                "title": source_issue.get("title"),
                "state": source_issue.get("state"),
                "html_url": src_html_url,
            })

    return {"merged_prs": merged_prs, "closed_issues": closed_issues}


def fetch_pr_bundle(api: GhApi, owner: str, repo: str, pr_number: int):
    """
    获取 PR 的完整信息包（PR详情、Issue评论、Review评论、提交记录、
    PR后续discussion、code review、code review下discussion、
    PR被mention的链接（仅限本仓库的merged pr和closed issue）、
    issue被mention的链接（仅限本仓库的merged pr和closed issue））
    """
    pr = call_api(api, api.pulls.get, owner=owner, repo=repo, pull_number=pr_number)
    if pr is None:
        return None

    issue_set = extract_resolved_issues(api, owner, repo, pr)
    # print(type(issue_set)) # list
    # print(f'for pr {pr_number}, issue_set: {issue_set}')

    issue_comments = []
    is_issue_mentioned = []

    for i in issue_set:
        issue_comments.extend(
            list(get_all_loop(api, api.issues.list_comments, owner=owner, repo=repo, issue_number=i, quiet=True))
        )
        is_issue_mentioned.append({"issue_number": i, "mentions": get_mentioned_issues_and_prs(api, owner, repo, i, target_type="issue")})

    # code review下的comments
    review_comments = list(get_all_loop(api, api.pulls.list_reviews, owner=owner, repo=repo, pull_number=pr_number))
    # pr number and issue number are not the same. if we want to get pr comments, just use pr number with issue liks.
    commits = list(get_all_loop(api, api.pulls.list_commits, owner=owner, repo=repo, pull_number=pr_number))
    pr_comments = list(get_all_loop(api, api.issues.list_comments, owner=owner, repo=repo, issue_number=pr_number, quiet=True))

    is_pr_mentioned = get_mentioned_issues_and_prs(api, owner, repo, pr_number, target_type="pr")
    files = list(get_all_loop(api, api.pulls.list_files, owner=owner, repo=repo, pull_number=pr_number))

    return {
        "pr": pr,
        "issue_comments": issue_comments,
        "pr_comments": pr_comments,
        "review_comments": review_comments,
        "commits": commits,
        "is_pr_mentioned": is_pr_mentioned,
        "is_issue_mentioned": is_issue_mentioned,
        "files": files,
        "issue_number": issue_set,
    }


def extract_text_list(items):
    """提取对象列表中的 body 文本"""
    texts = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                body = item.get("body")
            else:
                body = getattr(item, "body", None)
            if isinstance(body, str):
                texts.append(body)
    return texts

def build_pr_description(bundle: dict):
    """构建 PR 讨论内容（合并 PR 描述和所有评论）"""
    pr = bundle.get("pr")
    parts = []
    # 添加 PR 描述
    if isinstance(pr, dict):
        pr_title = pr.get("title")
        pr_body = pr.get("body")
    else:
        pr_title = getattr(pr, "title", None)
        pr_body = getattr(pr, "body", None)
    if isinstance(pr_title, str) and isinstance(pr_body, str):
        parts.append("PR Title: " + pr_title + ". \n\n" + "PR Description: " + pr_body)
    
    return "\n\n".join([p for p in parts if p.strip()])

def build_commit_messages(bundle: dict):
    """构建合并后的提交信息"""
    commits = bundle.get("commits")
    messages = []
    if isinstance(commits, list):
        for c in commits:
            message = None
            if isinstance(c, dict):
                commit = c.get("commit")
                if isinstance(commit, dict):
                    message = commit.get("message")
                if message is None:
                    message = c.get("message")
            else:
                commit = getattr(c, "commit", None)
                if commit is not None:
                    message = getattr(commit, "message", None)
                if message is None:
                    message = getattr(c, "message", None)
            if isinstance(message, str):
                messages.append(message)
    return messages



def main():
    parser = argparse.ArgumentParser(description="从 GitHub 获取 PR 信息并构建数据集")
    parser.add_argument("--input", required=True, help="输入文件路径 (json/jsonl)")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--repo-name", default=None, help="强制指定仓库名")
    parser.add_argument("--token", default=None, help="GitHub Token")
    parser.add_argument("--no-fetch", action="store_true", help="跳过 GitHub 数据获取")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(1)

    # 读取数据
    data = parse_json_or_jsonl(args.input)
    if isinstance(data, dict):
        data = [data]

    token = args.token or os.environ.get("GITHUB_TOKEN")
    api = GhApi(token=token)

    # 处理并流式追加写入输出文件（每构造一条写一条，不清空已有内容）
    is_jsonl = args.output.endswith(".jsonl")
    if not is_jsonl:
        print(f"Warning: output is not .jsonl ({args.output}), appending as JSONL lines.")

    with open(args.output, "a", encoding="utf-8") as f:

        for item in data:
            instance_id = item.get("instance_id")
            if not isinstance(instance_id, str):
                item.setdefault("pr_discussion", "")
                item.setdefault("commit_message", "")
            else:
                # 解析仓库和 PR 信息
                repo_name, pr_number = parse_instance_id(instance_id, args.repo_name)
                if args.no_fetch or repo_name is None or pr_number is None:
                    item.setdefault("pr_discussion", "")
                    item.setdefault("commit_message", "")
                else:
                    owner, name = repo_name.split("/")

                    # 获取 GitHub 数据
                    bundle = fetch_pr_bundle(api, owner, name, pr_number)
                    if not bundle:
                        item.setdefault("pr_discussion", "")
                        item.setdefault("commit_message", "")
                    else:
                        # 填充数据
                        item["pr_description"] = build_pr_description(bundle)
                        item["review_comments"] = extract_text_list(bundle.get("review_comments"))
                        item["pr_comments"] = extract_text_list(bundle.get("pr_comments"))
                        item["commit_message"] = build_commit_messages(bundle)
                        item['new_issue_comments'] = extract_text_list(bundle['issue_comments'])
                        item['is_pr_mentioned'] = bundle['is_pr_mentioned']
                        item['is_issue_mentioned'] = bundle['is_issue_mentioned']
                        item['files'] = bundle['files']
                        item['issue_number'] = bundle['issue_number']

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            f.flush()


if __name__ == "__main__":
    main()
