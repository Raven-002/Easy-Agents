"""Main module."""
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ai_cr.utils.gitlab_utils import GitlabMergeRequestApi


class CodeReviewAbstractComment(ABC):
    @abstractmethod
    def print_comment(self, console: Console):
        raise NotImplementedError

    @abstractmethod
    def submit_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        raise NotImplementedError


@dataclass
class CodeReviewGeneralComment(CodeReviewAbstractComment):
    comment: str

    def print_comment(self, console: Console):
        console.print(Panel(Markdown(self.comment), title="General comment"))

    def submit_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        gitlab_mr_api.create_merge_request_comment(self.comment)


@dataclass
class CodeReviewGeneralThread(CodeReviewAbstractComment):
    comment: str

    def print_comment(self, console: Console):
        console.print(Panel(Markdown(self.comment), title="General thread"))

    def submit_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        gitlab_mr_api.create_merge_request_thread(self.comment)


@dataclass
class CodeReviewFileThread(CodeReviewAbstractComment):
    filename: str
    comment: str

    def print_comment(self, console: Console):
        console.print(Panel(Markdown(self.comment), title=f"{self.filename}"))

    def submit_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        gitlab_mr_api.create_merge_request_file_thread(self.filename, self.comment)


@dataclass
class CodeReviewThread(CodeReviewAbstractComment):
    filename: str
    line_number: int
    comment: str

    def print_comment(self, console: Console):
        console.print(Panel(Markdown(self.comment), title=f"{self.filename}:{self.line_number}"))

    def submit_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        gitlab_mr_api.create_merge_request_line_thread(self.filename, self.line_number, self.comment)


@dataclass
class CodeReview:
    reviewer: str
    comments: list[CodeReviewAbstractComment]

    def print_comments(self, console: Console):
        for comment in self.comments:
            comment.print_comment(console)

    async def submit_comments_to_gitlab(self, gitlab_mr_api: GitlabMergeRequestApi):
        await asyncio.gather(*(comment.submit_to_gitlab(gitlab_mr_api) for comment in self.comments),
                             return_exceptions=True)
