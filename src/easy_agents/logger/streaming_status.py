# import time
#
# from rich.console import Console, Group
# from rich.live import Live
# from rich.spinner import Spinner
# from rich.text import Text
#
#
# class StreamingSession:
#     """
#     Context manager for token streaming with spinner at bottom.
#
#     Layout:
#     - Streaming tokens appear on the top line
#     - Spinner/status appears on the bottom line
#
#     Usage:
#         with StreamingSession(console, "Processing...") as session:
#             for token in tokens:
#                 session.stream_token(token)
#     """
#
#     def __init__(self, console: Console, status_text: str = "Streaming..."):
#         self._console = console
#         self.status_text = status_text
#         self.streamed_text = Text()
#         self.live = None
#
#     def __enter__(self):
#         """Start the streaming session."""
#         self.streamed_text = Text()
#
#         # Start Live display
#         self.live = Live(
#             self._get_renderable(),
#             console=self.console,
#             refresh_per_second=10,
#             transient=True,  # Remove when done
#         )
#         self.live.start()
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Stop streaming and finalize output."""
#         if self.live:
#             self.live.stop()
#             self.live = None
#
#         # Print the final streamed content permanently
#         self.console.print(self.streamed_text)
#
#         return False  # Don't suppress exceptions
#
#     def stream_token(self, token: str):
#         """Add a token to the current streaming session."""
#         self.streamed_text.append(token)
#         if self.live:
#             self.live.update(self._get_renderable())
#
#     def update_status(self, new_status: str):
#         """Update the status message."""
#         self.status_text = new_status
#         if self.live:
#             self.live.update(self._get_renderable())
#
#     @property
#     def console(self) -> Console:
#         return self._console
#
#     def _get_renderable(self):
#         """Create a display with tokens on top, spinner on the bottom."""
#         spinner = Spinner("dots", text=self.status_text)
#
#         # Tokens on top, spinner on bottom
#         return Group(self.streamed_text, spinner)
#
#
# # Example usage demonstrating the workflow
# def simulate_llm_workflow():
#     console = Console()
#
#     # Initial log
#     console.print("[bold blue]Starting LLM interaction...[/bold blue]")
#     console.rule("Session 1")
#
#     # First streaming session
#     tokens_1 = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]
#
#     with StreamingSession(console, "Running LLM (Query 1)...") as session:
#         for token in tokens_1:
#             session.stream_token(token)
#             time.sleep(0.1)
#
#     # Regular logs between streaming
#     console.print("\n[yellow]⚠ Processing intermediate step...[/yellow]")
#     console.print("[dim]Debug: Token count = 10[/dim]")
#
#     # Separator
#     console.rule("Session 2")
#
#     # Second streaming session with dynamic status
#     tokens_2 = ["Now", " streaming", " a", " second", " response", " from", " the", " LLM", "!"]
#
#     with StreamingSession(console, "Running LLM (Query 2)...") as session:
#         for i, token in enumerate(tokens_2):
#             # Update status dynamically
#             session.update_status(f"Running LLM (Query 2)... [{i + 1}/{len(tokens_2)}]")
#             session.stream_token(token)
#             time.sleep(0.1)
#
#     # More logs
#     console.print("\n[green]✓ Analysis complete[/green]")
#     console.print("[dim]Final token count = 19[/dim]")
#
#     # Another separator
#     console.rule("Session 3")
#
#     # Third streaming session with manual interruption
#     tokens_3_part1 = ["This", " is", " a"]
#     tokens_3_part2 = [" longer", " streaming", " session", " with", " an", " interruption", "!"]
#
#     with StreamingSession(console, "Running LLM (Query 3)...") as session:
#         for token in tokens_3_part1:
#             session.stream_token(token)
#             time.sleep(0.1)
#
#     # Log message between (streaming stopped automatically)
#     console.print("[red]⚠ Warning: Context window approaching limit[/red]")
#
#     # Continue with a new session
#     with StreamingSession(console, "Running LLM (Query 3, continued)...") as session:
#         for token in tokens_3_part2:
#             session.stream_token(token)
#             time.sleep(0.1)
#
#     # Final output
#     console.print("\n[bold green]✓ All sessions complete![/bold green]")
#
#     # Demonstrate multi-line streaming
#     console.rule("Session 4 - Multi-line")
#
#     tokens_4 = [
#         "This",
#         " is",
#         " a",
#         " multi",
#         "-",
#         "line",
#         " response",
#         ".\n\n",
#         "It",
#         " contains",
#         " multiple",
#         " paragraphs",
#         ".\n\n",
#         "And",
#         " the",
#         " spinner",
#         " stays",
#         " at",
#         " the",
#         " bottom",
#         "!",
#     ]
#
#     with StreamingSession(console, "Generating multi-line response...") as session:
#         for token in tokens_4:
#             session.stream_token(token)
#             time.sleep(0.08)
#
#
# if __name__ == "__main__":
#     simulate_llm_workflow()
