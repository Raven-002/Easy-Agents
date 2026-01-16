class GitlabMergeRequestApi:
    async def create_merge_request_comment(self, comment: str) -> None:
        raise NotImplementedError

    async def create_merge_request_thread(self, comment: str) -> None:
        raise NotImplementedError

    async def create_merge_request_file_thread(self, filename: str, comment: str) -> None:
        raise NotImplementedError

    async def create_merge_request_line_thread(self, filename: str, line_number: int, comment: str) -> None:
        raise NotImplementedError
