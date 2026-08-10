import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from dotenv import load_dotenv

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.gemini_wrapper import GeminiChatWrapper
from src.llm.persona_manager import PersonaManager
from src.ingestion.lore_processor import LoreProcessor
from src.ingestion.script_parser import ScriptParser

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
        self.retriever = HybridRetriever(self.vs_manager)

    def _bootstrap_data(self):
        console.print("[bold yellow]Holocron energy low. Initializing data core...[/bold yellow]")
        lore_proc = LoreProcessor()
        script_parser = ScriptParser()

        docs = []
        missing = []

        # Ingest Lore
        lore_file = "data/raw/lore.json"
        if os.path.exists(lore_file):
            docs.extend(lore_proc.process_file(lore_file))
        else:
            missing.append(lore_file)

        # Ingest Scripts. These live in third-party corpora fetched by
        # scripts/fetch_corpora.py -- absent them the index is lore-only and the
        # retriever's dialogue half is silently empty, so say so loudly.
        ot_script = "data/raw/star-wars-scripts/Text_files/EpisodeIV_dialogues.txt"
        if os.path.exists(ot_script):
            docs.extend(script_parser.parse_tab_txt(ot_script))
        else:
            missing.append(ot_script)

        prequel_script = "data/raw/prequel-csv/star_wars_1_data.csv"
        if os.path.exists(prequel_script):
            docs.extend(script_parser.parse_csv(prequel_script, sep=";", char_col="from", text_col="text"))
        else:
            missing.append(prequel_script)

        dialogue_count = sum(1 for d in docs if d.metadata.get("type") == "dialogue")

        if missing:
            console.print("[bold yellow]Missing corpora:[/bold yellow]")
            for path in missing:
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

        self.vs_manager.add_documents(docs)
        self.vs_manager.save()
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
                query = console.input(f"\n[bold white]You:[/bold white] ")
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
        welcome_text = """
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
