import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from src.ingestion.pipeline import build_index, collect_documents
from src.llm.gemini_wrapper import GeminiChatWrapper
from src.llm.persona_manager import PersonaManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStoreManager

load_dotenv()

console = Console()

class SithHolocronCLI:
    def __init__(self):
        self.pm = PersonaManager()
        self.vs_manager = VectorStoreManager(persist_directory="data/vector_store")

        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[bold red]ERROR:[/bold red] GOOGLE_API_KEY not found in environment.")
            sys.exit(1)
        self.llm = GeminiChatWrapper(api_key=api_key)

        # Initialize Retriever
        if not self.vs_manager.load():
            self._bootstrap_data()
        self.retriever = HybridRetriever(self.vs_manager, persona_manager=self.pm)

    def _bootstrap_data(self):
        console.print("[bold yellow]Holocron energy low. Initializing data core...[/bold yellow]")

        # Corpus discovery and the build itself live in src.ingestion.pipeline so
        # that scripts/build_index.py can produce an identical index without a
        # TTY or an API key. This method is now only the console reporting.
        report = collect_documents()
        docs = report.documents
        dialogue_count = report.dialogue_count

        if report.missing:
            console.print("[bold yellow]Missing corpora:[/bold yellow]")
            for path in report.missing:
                console.print(f"  - {path}")
            console.print(
                "[bold yellow]Run [white]python scripts/fetch_corpora.py[/white] "
                "to fetch the screenplay corpora.[/bold yellow]"
            )

        if not docs:
            console.print("[bold red]ERROR:[/bold red] No raw data found to index. Cannot continue.")
            sys.exit(1)

        if dialogue_count == 0:
            console.print(
                "[bold red]WARNING:[/bold red] Indexing lore only -- no dialogue documents. "
                "Persona voice matching will find nothing."
            )

        build_index(docs, vs_manager=self.vs_manager)
        console.print(
            f"[bold green]Data core stabilized.[/bold green] "
            f"{len(docs)} documents ({dialogue_count} dialogue)."
        )

    def run(self):
        self.display_welcome()

        # Persona selection
        options = list(self.pm.personas.keys())
        console.print("\n[bold cyan]Select a Persona to commune with:[/bold cyan]")
        for i, opt in enumerate(options):
            console.print(f" {i+1}. {self.pm.personas[opt]['name']}")

        choice = console.input("\n[bold white]Select number (default 1): [/bold white]")
        idx = int(choice) - 1 if choice.isdigit() and 0 < int(choice) <= len(options) else 0
        persona_key = options[idx]
        persona_name = self.pm.personas[persona_key]['name']

        console.print(Panel(f"Communing with [bold red]{persona_name}[/bold red]...", border_style="red"))

        while True:
            try:
                query = console.input("\n[bold white]You:[/bold white] ")
                if query.lower() in ["exit", "quit", "bye"]:
                    console.print("[italic red]The Holocron fades to black...[/italic red]")
                    break

                # RAG Cycle
                docs = self.retriever.retrieve(query, character=persona_key, k=4)
                context = self.pm.format_context(docs)
                system_prompt = self.pm.get_system_prompt(persona_key)

                # Streaming Response
                console.print(f"\n[bold red]{persona_name}:[/bold red] ", end="")
                full_response = ""
                with Live(Text(""), refresh_per_second=15, console=console) as live:
                    for chunk in self.llm.stream_chat(system_prompt, query, context):
                        full_response += chunk
                        live.update(Text(full_response, style="red"))
                print() # New line after stream

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"\n[bold red]Disturbance in the Force:[/bold red] {e}")

    def display_welcome(self):
        # Raw string: the banner contains \_ sequences, which are not valid
        # escapes. Python currently emits a SyntaxWarning and leaves them as
        # literal backslashes; a future version makes that an error.
        welcome_text = r"""
    .----------------.  .----------------.  .----------------.  .----------------.
    | .--------------. || .--------------. || .--------------. || .--------------. |
    | |    _______   | || |     _____    | || |  _________   | || |  ____  ____  | |
    | |   /  ___  |  | || |    |_   _|   | || | |  _   _  |  | || | |_   ||   _| | |
    | |  |  (__ \_|  | || |      | |     | || | |_/ | | \_|  | || |   | |__| |   | |
    | |   '.___\_    | || |      | |     | || |     | |      | || |   |  __  |   | |
    | |  |\____)  |  | || |     _| |_    | || |    _| |_     | || |  _| |  | |_  | |
    | |  |_________|  | || |    |_____|   | || |   |_____|    | || | |____||____| | |
    | |              | || |              | || |              | || |              | |
    | '--------------' || '--------------' || '--------------' || '--------------' |
    '----------------'  '----------------'  '----------------'  '----------------'
        """
        console.print(welcome_text, style="red")
        console.print(Panel("SITH HOLOCRON: CHRONICLE OF THE DARK SIDE", style="bold red", border_style="red"))

if __name__ == "__main__":
    cli = SithHolocronCLI()
    cli.run()
