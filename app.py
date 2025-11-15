import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime
import threading
from queue import Queue
from pdf_handler import PDFHandler
from openrouter_client import OpenRouterClient
from rag_handler import RAGHandler

class AILMApp:
    def __init__(self):
        # Configuration
        self.config_file = "config.json"
        self.pdf_data_file = "pdf_knowledge_base.json"
        self.config = self.load_config()
        self.pdf_knowledge = self.load_pdf_knowledge()

        # Initialize handlers
        self.pdf_handler = PDFHandler()
        self.openrouter_client = OpenRouterClient()
        self.rag_handler = RAGHandler()

        # Load RAG embeddings
        self.rag_handler.load_embeddings()

        # Chat history
        self.chat_history = []

        # Context tracking
        self.last_sent_context = ""
        self.context_window = None

        # Threading for multi-LLM requests
        self.response_queue = Queue()
        self.active_threads = []

        # Create main window
        self.root = ctk.CTk()
        self.root.title("AI Chat Assistant")
        self.root.geometry("900x700")

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create tabview
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self.tab_chat = self.tabview.add("Chat")
        self.tab_llm = self.tabview.add("LLM")
        self.tab_pdf = self.tabview.add("Base de Connaissances PDF")
        self.tab_config = self.tabview.add("Configuration")

        # Setup each tab
        self.setup_chat_tab()
        self.setup_llm_tab()
        self.setup_pdf_tab()
        self.setup_config_tab()

    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "api_key": "",
            "models": [
                {"name": "anthropic/claude-3.5-sonnet", "enabled": True},
                {"name": "openai/gpt-4-turbo", "enabled": False}
            ],
            "use_rag": False,
            "rag_chunk_size": 500,
            "rag_chunk_overlap": 50,
            "rag_top_k": 5,
            "use_structured_output": False
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                    # Backward compatibility: migrate old "model" field to "models" list
                    if "model" in config and "models" not in config:
                        old_model = config.pop("model")
                        config["models"] = [{"name": old_model, "enabled": True}]

                    # Ensure models field exists
                    if "models" not in config:
                        config["models"] = default_config["models"]

                    return config
            except:
                pass

        return default_config

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_enabled_models(self):
        """Get list of enabled model names"""
        return [model["name"] for model in self.config.get("models", []) if model.get("enabled", False)]

    def add_model(self, model_name, enabled=True):
        """Add a new model to the list"""
        if "models" not in self.config:
            self.config["models"] = []

        # Check if model already exists
        for model in self.config["models"]:
            if model["name"] == model_name:
                return False  # Model already exists

        self.config["models"].append({"name": model_name, "enabled": enabled})
        self.save_config()
        return True

    def remove_model(self, model_name):
        """Remove a model from the list"""
        if "models" not in self.config:
            return False

        original_length = len(self.config["models"])
        self.config["models"] = [m for m in self.config["models"] if m["name"] != model_name]

        if len(self.config["models"]) < original_length:
            self.save_config()
            return True
        return False

    def toggle_model(self, model_name):
        """Toggle a model's enabled status"""
        if "models" not in self.config:
            return False

        for model in self.config["models"]:
            if model["name"] == model_name:
                model["enabled"] = not model.get("enabled", False)
                self.save_config()
                return True
        return False

    def load_pdf_knowledge(self):
        """Load PDF knowledge base from file"""
        if os.path.exists(self.pdf_data_file):
            try:
                with open(self.pdf_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def save_pdf_knowledge(self):
        """Save PDF knowledge base to file"""
        with open(self.pdf_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.pdf_knowledge, f, indent=2, ensure_ascii=False)

    def setup_chat_tab(self):
        """Setup the chat interface tab"""
        # Context usage frame
        context_frame = ctk.CTkFrame(self.tab_chat)
        context_frame.pack(fill="x", padx=10, pady=(10, 0))

        # Context usage label
        self.context_label = ctk.CTkLabel(
            context_frame,
            text="Utilisation du contexte: -- / --",
            font=("Arial", 11)
        )
        self.context_label.pack(side="left", padx=10, pady=5)

        # Context usage progress bar
        self.context_progress = ctk.CTkProgressBar(
            context_frame,
            width=300,
            height=15
        )
        self.context_progress.pack(side="left", padx=10, pady=5)
        self.context_progress.set(0)

        # Percentage label
        self.context_percentage = ctk.CTkLabel(
            context_frame,
            text="0%",
            font=("Arial", 11, "bold")
        )
        self.context_percentage.pack(side="left", padx=5, pady=5)

        # View context button
        self.view_context_button = ctk.CTkButton(
            context_frame,
            text="Voir Contexte",
            command=self.show_context_window,
            width=120,
            height=30,
            fg_color="#2B5278"
        )
        self.view_context_button.pack(side="right", padx=10, pady=5)

        # Chat display area
        self.chat_display = ctk.CTkTextbox(
            self.tab_chat,
            wrap="word",
            state="disabled",
            font=("Arial", 12)
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)

        # Input frame
        input_frame = ctk.CTkFrame(self.tab_chat)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Message input
        self.message_input = ctk.CTkTextbox(
            input_frame,
            height=80,
            font=("Arial", 12)
        )
        self.message_input.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Button frame
        button_frame = ctk.CTkFrame(input_frame)
        button_frame.pack(side="right", fill="y")

        # Send button
        self.send_button = ctk.CTkButton(
            button_frame,
            text="Envoyer",
            command=self.send_message,
            width=100,
            height=40
        )
        self.send_button.pack(pady=(0, 5))

        # Clear chat button
        self.clear_button = ctk.CTkButton(
            button_frame,
            text="Effacer",
            command=self.clear_chat,
            width=100,
            height=30,
            fg_color="gray"
        )
        self.clear_button.pack()

        # Bind Enter key to send message
        self.message_input.bind("<Control-Return>", lambda e: self.send_message())

    def setup_llm_tab(self):
        """Setup the LLM monitoring tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_llm,
            text="Monitoring des LLMs",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        # Description
        desc_label = ctk.CTkLabel(
            self.tab_llm,
            text="Surveillance en temps réel de l'état et des performances de chaque modèle LLM",
            font=("Arial", 11),
            text_color="gray"
        )
        desc_label.pack(pady=(0, 10))

        # Scrollable frame for LLM cards
        self.llm_cards_frame = ctk.CTkScrollableFrame(
            self.tab_llm,
            label_text="Modèles Actifs"
        )
        self.llm_cards_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Dictionary to store LLM status widgets
        self.llm_status_widgets = {}

        # Initialize LLM cards
        self.update_llm_cards()

    def update_llm_cards(self):
        """Update the LLM monitoring cards"""
        # Clear existing widgets
        for widget in self.llm_cards_frame.winfo_children():
            widget.destroy()

        self.llm_status_widgets.clear()

        # Get enabled models
        enabled_models = self.get_enabled_models()

        if not enabled_models:
            # No models enabled
            no_models_label = ctk.CTkLabel(
                self.llm_cards_frame,
                text="Aucun modèle activé\n\nActivez des modèles dans l'onglet Configuration",
                font=("Arial", 12),
                text_color="gray"
            )
            no_models_label.pack(pady=50)
            return

        # Create a card for each enabled model
        for model_name in enabled_models:
            self.create_llm_card(model_name)

    def create_llm_card(self, model_name):
        """Create a monitoring card for a specific LLM"""
        # Card frame
        card_frame = ctk.CTkFrame(self.llm_cards_frame)
        card_frame.pack(fill="x", padx=5, pady=5)

        # Header frame
        header_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)

        # Model name
        model_label = ctk.CTkLabel(
            header_frame,
            text=f"🤖 {model_name}",
            font=("Arial", 13, "bold"),
            anchor="w"
        )
        model_label.pack(side="left", padx=5)

        # Status indicator
        status_label = ctk.CTkLabel(
            header_frame,
            text="⚪ Inactif",
            font=("Arial", 11),
            text_color="gray"
        )
        status_label.pack(side="right", padx=5)

        # Context usage frame
        context_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        context_frame.pack(fill="x", padx=10, pady=5)

        context_label = ctk.CTkLabel(
            context_frame,
            text="Contexte: -- / --",
            font=("Arial", 10)
        )
        context_label.pack(side="left", padx=5)

        context_progress = ctk.CTkProgressBar(
            context_frame,
            width=200,
            height=10
        )
        context_progress.pack(side="left", padx=5)
        context_progress.set(0)

        context_percentage = ctk.CTkLabel(
            context_frame,
            text="0%",
            font=("Arial", 10)
        )
        context_percentage.pack(side="left", padx=5)

        # Response/Error frame
        response_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        response_frame.pack(fill="x", padx=10, pady=5)

        response_label = ctk.CTkLabel(
            response_frame,
            text="Statut: En attente de requête...",
            font=("Arial", 10),
            text_color="gray",
            anchor="w",
            justify="left"
        )
        response_label.pack(fill="x", padx=5, pady=2)

        # Timestamp
        timestamp_label = ctk.CTkLabel(
            card_frame,
            text="",
            font=("Arial", 9),
            text_color="gray",
            anchor="w"
        )
        timestamp_label.pack(fill="x", padx=15, pady=(0, 5))

        # Store widgets for this model
        self.llm_status_widgets[model_name] = {
            "card_frame": card_frame,
            "status_label": status_label,
            "context_label": context_label,
            "context_progress": context_progress,
            "context_percentage": context_percentage,
            "response_label": response_label,
            "timestamp_label": timestamp_label
        }

    def update_llm_status(self, model_name, status, data=None):
        """
        Update the status of a specific LLM

        Args:
            model_name: Name of the model
            status: Status string ("idle", "processing", "completed", "error")
            data: Dictionary with optional keys:
                - response: Response text preview
                - error: Error message
                - usage: Token usage dict
                - timestamp: Timestamp string
        """
        if model_name not in self.llm_status_widgets:
            return

        widgets = self.llm_status_widgets[model_name]
        data = data or {}

        # Update status indicator
        status_colors = {
            "idle": ("⚪ Inactif", "gray"),
            "processing": ("🟡 En cours...", "#FFA500"),
            "completed": ("🟢 Complété", "green"),
            "error": ("🔴 Erreur", "red")
        }

        status_text, status_color = status_colors.get(status, ("⚪ Inactif", "gray"))
        widgets["status_label"].configure(text=status_text, text_color=status_color)

        # Update context usage if available
        if "usage" in data:
            usage = data["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            context_limit = data.get("context_limit", 8192)
            percentage = (prompt_tokens / context_limit) * 100 if context_limit > 0 else 0

            widgets["context_label"].configure(text=f"Contexte: {prompt_tokens:,} / {context_limit:,}")
            widgets["context_progress"].set(min(prompt_tokens / context_limit, 1.0) if context_limit > 0 else 0)
            widgets["context_percentage"].configure(text=f"{percentage:.1f}%")

            # Color code percentage
            if percentage < 50:
                color = "green"
            elif percentage < 80:
                color = "orange"
            else:
                color = "red"
            widgets["context_percentage"].configure(text_color=color)

        # Update response/error message
        if "error" in data:
            widgets["response_label"].configure(
                text=f"Erreur: {data['error'][:200]}...",
                text_color="red"
            )
        elif "response" in data:
            preview = data["response"][:150]
            if len(data["response"]) > 150:
                preview += "..."
            widgets["response_label"].configure(
                text=f"Réponse: {preview}",
                text_color="white"
            )
        elif status == "processing":
            widgets["response_label"].configure(
                text="Statut: Envoi de la requête au modèle...",
                text_color="gray"
            )

        # Update timestamp
        if "timestamp" in data:
            widgets["timestamp_label"].configure(text=f"Dernière mise à jour: {data['timestamp']}")

    def setup_pdf_tab(self):
        """Setup the PDF management tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_pdf,
            text="Gestion de la Base de Connaissances PDF",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        # RAG toggle frame
        rag_frame = ctk.CTkFrame(self.tab_pdf)
        rag_frame.pack(fill="x", padx=20, pady=10)

        # RAG label
        rag_label = ctk.CTkLabel(
            rag_frame,
            text="Mode RAG (Retrieval-Augmented Generation):",
            font=("Arial", 13, "bold")
        )
        rag_label.pack(side="left", padx=10, pady=10)

        # RAG switch
        self.rag_switch = ctk.CTkSwitch(
            rag_frame,
            text="Activé" if self.config.get("use_rag", False) else "Désactivé",
            command=self.toggle_rag,
            font=("Arial", 12)
        )
        if self.config.get("use_rag", False):
            self.rag_switch.select()
        self.rag_switch.pack(side="left", padx=10, pady=10)

        # RAG info label
        self.rag_info_label = ctk.CTkLabel(
            rag_frame,
            text="",
            font=("Arial", 10),
            text_color="gray"
        )
        self.rag_info_label.pack(side="left", padx=10, pady=10)
        self.update_rag_info()

        # RAG description
        rag_desc = ctk.CTkLabel(
            self.tab_pdf,
            text="Le mode RAG utilise la recherche sémantique pour trouver les passages pertinents.\n"
                 "Recommandé pour les PDFs volumineux (>100 pages).",
            font=("Arial", 10),
            text_color="gray"
        )
        rag_desc.pack(pady=(0, 10))

        # RAG Parameters frame
        rag_params_frame = ctk.CTkFrame(self.tab_pdf)
        rag_params_frame.pack(fill="x", padx=20, pady=10)

        # Title for parameters
        params_title = ctk.CTkLabel(
            rag_params_frame,
            text="Paramètres RAG",
            font=("Arial", 12, "bold")
        )
        params_title.grid(row=0, column=0, columnspan=3, pady=(10, 15), padx=10, sticky="w")

        # Chunk Size
        chunk_size_label = ctk.CTkLabel(
            rag_params_frame,
            text="Taille des chunks (mots):",
            font=("Arial", 11)
        )
        chunk_size_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.chunk_size_entry = ctk.CTkEntry(
            rag_params_frame,
            width=80,
            placeholder_text="500"
        )
        self.chunk_size_entry.insert(0, str(self.config.get("rag_chunk_size", 500)))
        self.chunk_size_entry.grid(row=1, column=1, padx=5, pady=5)

        chunk_size_info = ctk.CTkLabel(
            rag_params_frame,
            text="Nombre de mots par segment (défaut: 500)",
            font=("Arial", 9),
            text_color="gray"
        )
        chunk_size_info.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # Chunk Overlap
        overlap_label = ctk.CTkLabel(
            rag_params_frame,
            text="Chevauchement (mots):",
            font=("Arial", 11)
        )
        overlap_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.chunk_overlap_entry = ctk.CTkEntry(
            rag_params_frame,
            width=80,
            placeholder_text="50"
        )
        self.chunk_overlap_entry.insert(0, str(self.config.get("rag_chunk_overlap", 50)))
        self.chunk_overlap_entry.grid(row=2, column=1, padx=5, pady=5)

        overlap_info = ctk.CTkLabel(
            rag_params_frame,
            text="Mots en commun entre chunks (défaut: 50)",
            font=("Arial", 9),
            text_color="gray"
        )
        overlap_info.grid(row=2, column=2, padx=10, pady=5, sticky="w")

        # Top K
        topk_label = ctk.CTkLabel(
            rag_params_frame,
            text="Nombre de résultats:",
            font=("Arial", 11)
        )
        topk_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.topk_entry = ctk.CTkEntry(
            rag_params_frame,
            width=80,
            placeholder_text="5"
        )
        self.topk_entry.insert(0, str(self.config.get("rag_top_k", 5)))
        self.topk_entry.grid(row=3, column=1, padx=5, pady=5)

        topk_info = ctk.CTkLabel(
            rag_params_frame,
            text="Chunks les plus pertinents à retourner (défaut: 5)",
            font=("Arial", 9),
            text_color="gray"
        )
        topk_info.grid(row=3, column=2, padx=10, pady=5, sticky="w")

        # Save RAG parameters button
        save_rag_params_btn = ctk.CTkButton(
            rag_params_frame,
            text="Sauvegarder les paramètres",
            command=self.save_rag_parameters,
            width=200,
            height=30
        )
        save_rag_params_btn.grid(row=4, column=0, columnspan=3, pady=(15, 10), padx=10)

        # Upload button
        upload_btn = ctk.CTkButton(
            self.tab_pdf,
            text="Ajouter un PDF",
            command=self.upload_pdf,
            width=200,
            height=40
        )
        upload_btn.pack(pady=10)

        # PDF list frame
        list_frame = ctk.CTkFrame(self.tab_pdf)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # List label
        list_label = ctk.CTkLabel(
            list_frame,
            text="PDFs dans la base de connaissances:",
            font=("Arial", 14, "bold")
        )
        list_label.pack(pady=5)

        # Scrollable frame for PDF list
        self.pdf_list_frame = ctk.CTkScrollableFrame(list_frame)
        self.pdf_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Update PDF list display
        self.update_pdf_list()

    def setup_config_tab(self):
        """Setup the configuration tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_config,
            text="Configuration OpenRouter",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=20)

        # API Key section
        api_frame = ctk.CTkFrame(self.tab_config)
        api_frame.pack(fill="x", padx=20, pady=10)

        api_label = ctk.CTkLabel(
            api_frame,
            text="Clé API OpenRouter:",
            font=("Arial", 14)
        )
        api_label.pack(pady=5)

        self.api_key_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="Entrez votre clé API",
            width=400,
            show="*"
        )
        self.api_key_entry.pack(pady=5)
        if self.config.get("api_key"):
            self.api_key_entry.insert(0, self.config["api_key"])

        # Models section
        models_frame = ctk.CTkFrame(self.tab_config)
        models_frame.pack(fill="both", expand=True, padx=20, pady=10)

        models_label = ctk.CTkLabel(
            models_frame,
            text="Modèles LLM:",
            font=("Arial", 14, "bold")
        )
        models_label.pack(pady=5)

        # Info label
        info_label = ctk.CTkLabel(
            models_frame,
            text="Gérez vos modèles LLM - Les modèles activés recevront la même question en parallèle",
            font=("Arial", 10),
            text_color="gray"
        )
        info_label.pack(pady=(0, 10))

        # Scrollable frame for models list
        self.models_list_frame = ctk.CTkScrollableFrame(
            models_frame,
            height=200
        )
        self.models_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Add model section
        add_model_frame = ctk.CTkFrame(models_frame, fg_color="transparent")
        add_model_frame.pack(fill="x", padx=5, pady=10)

        add_label = ctk.CTkLabel(
            add_model_frame,
            text="Ajouter un modèle:",
            font=("Arial", 11)
        )
        add_label.pack(side="left", padx=5)

        self.new_model_entry = ctk.CTkEntry(
            add_model_frame,
            placeholder_text="Ex: anthropic/claude-3.5-sonnet",
            width=300
        )
        self.new_model_entry.pack(side="left", padx=5)

        add_btn = ctk.CTkButton(
            add_model_frame,
            text="Ajouter",
            command=self.add_model_to_list,
            width=100,
            height=30
        )
        add_btn.pack(side="left", padx=5)

        # Examples label
        examples_label = ctk.CTkLabel(
            models_frame,
            text="Exemples: anthropic/claude-3.5-sonnet, openai/gpt-4-turbo, google/gemini-pro",
            font=("Arial", 9),
            text_color="gray"
        )
        examples_label.pack(pady=(0, 5))

        # Initialize models list display
        self.update_models_list()

        # Structured Output section
        structured_frame = ctk.CTkFrame(self.tab_config)
        structured_frame.pack(fill="x", padx=20, pady=10)

        structured_label = ctk.CTkLabel(
            structured_frame,
            text="Structured Output (Sortie Structurée):",
            font=("Arial", 14, "bold")
        )
        structured_label.pack(pady=5)

        # Structured output switch
        self.structured_output_switch = ctk.CTkSwitch(
            structured_frame,
            text="Activé" if self.config.get("use_structured_output", False) else "Désactivé",
            command=self.toggle_structured_output,
            font=("Arial", 12)
        )
        if self.config.get("use_structured_output", False):
            self.structured_output_switch.select()
        self.structured_output_switch.pack(pady=5)

        # Structured output description
        structured_desc = ctk.CTkLabel(
            structured_frame,
            text="Format de réponse JSON:\n"
                 '{\n'
                 '  "explanation": "Explication détaillée",\n'
                 '  "final_answer": "Réponse finale",\n'
                 '  "final_answer_letter": "Lettre de la réponse (ex: A, B, C, D)"\n'
                 '}\n\n'
                 "Compatible avec: OpenAI GPT-4, GPT-4 Turbo, GPT-3.5 Turbo",
            font=("Arial", 10),
            text_color="gray",
            justify="left"
        )
        structured_desc.pack(pady=5)

        # Save button
        save_btn = ctk.CTkButton(
            self.tab_config,
            text="Sauvegarder la Configuration",
            command=self.save_configuration,
            width=200,
            height=40
        )
        save_btn.pack(pady=20)

        # Status label
        self.config_status_label = ctk.CTkLabel(
            self.tab_config,
            text="",
            font=("Arial", 12)
        )
        self.config_status_label.pack(pady=5)

    def llm_worker(self, model_name, message, context, api_key, use_structured_output):
        """
        Worker thread function to send a request to a single LLM

        Args:
            model_name: Name of the model
            message: User message
            context: Context from RAG/PDFs
            api_key: OpenRouter API key
            use_structured_output: Whether to use structured output
        """
        try:
            # Update status to processing
            self.response_queue.put({
                "type": "status",
                "model": model_name,
                "status": "processing",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Send request
            result = self.openrouter_client.send_message(
                message,
                api_key,
                model_name,
                context,
                self.chat_history,
                use_structured_output=use_structured_output
            )

            # Get context limit
            context_limit = self.openrouter_client.get_context_limit(
                model_name,
                api_key
            )

            # Send success response
            self.response_queue.put({
                "type": "response",
                "model": model_name,
                "status": "completed",
                "content": result['content'],
                "usage": result.get('usage', {}),
                "context_limit": context_limit,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        except Exception as e:
            # Send error response
            self.response_queue.put({
                "type": "error",
                "model": model_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    def process_response_queue(self):
        """Process responses from worker threads and update UI"""
        try:
            while not self.response_queue.empty():
                response = self.response_queue.get_nowait()

                if response["type"] == "status":
                    # Update LLM status
                    self.update_llm_status(
                        response["model"],
                        response["status"],
                        {"timestamp": response["timestamp"]}
                    )

                elif response["type"] == "response":
                    # Update LLM status with response
                    self.update_llm_status(
                        response["model"],
                        response["status"],
                        {
                            "response": response["content"],
                            "usage": response["usage"],
                            "context_limit": response["context_limit"],
                            "timestamp": response["timestamp"]
                        }
                    )

                    # Add response to chat
                    self.add_llm_response_to_chat(
                        response["model"],
                        response["content"]
                    )

                elif response["type"] == "error":
                    # Update LLM status with error
                    self.update_llm_status(
                        response["model"],
                        response["status"],
                        {
                            "error": response["error"],
                            "timestamp": response["timestamp"]
                        }
                    )

                    # Add error to chat
                    self.add_message_to_chat(
                        f"Erreur ({response['model']})",
                        response["error"]
                    )

        except:
            pass

        # Schedule next check
        self.root.after(100, self.process_response_queue)

    def add_llm_response_to_chat(self, model_name, content):
        """Add LLM response to chat display"""
        # Parse and format if structured output
        if self.config.get("use_structured_output", False):
            try:
                parsed_response = json.loads(content)
                formatted_response = self.format_structured_response(parsed_response)
                display_content = formatted_response
            except json.JSONDecodeError:
                display_content = content
        else:
            display_content = content

        # Add to chat with model name
        self.add_message_to_chat(f"Assistant ({model_name})", display_content)

    def send_message(self):
        """Send message to multiple LLMs in parallel"""
        message = self.message_input.get("1.0", "end-1c").strip()

        if not message:
            return

        # Check if API key is configured
        if not self.config.get("api_key"):
            messagebox.showerror(
                "Erreur",
                "Veuillez configurer votre clé API dans l'onglet Configuration"
            )
            return

        # Get enabled models
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            messagebox.showerror(
                "Erreur",
                "Aucun modèle activé. Veuillez activer au moins un modèle dans Configuration."
            )
            return

        # Clear input
        self.message_input.delete("1.0", "end")

        # Add user message to chat
        self.add_message_to_chat("Vous", message)

        # Disable send button while processing
        self.send_button.configure(state="disabled", text="En cours...")
        self.root.update()

        # Build context from PDF knowledge
        context = self.build_context(message)

        # Store the context that will be sent
        self.last_sent_context = context

        # Update context window if it's open
        self.update_context_window_if_open()

        # Update chat history with user message
        self.chat_history.append({"role": "user", "content": message})

        # Reset LLM statuses to idle
        for model_name in enabled_models:
            self.update_llm_status(model_name, "idle", {})

        # Start worker threads for each enabled model
        self.active_threads = []
        for model_name in enabled_models:
            thread = threading.Thread(
                target=self.llm_worker,
                args=(
                    model_name,
                    message,
                    context,
                    self.config["api_key"],
                    self.config.get("use_structured_output", False)
                ),
                daemon=True
            )
            thread.start()
            self.active_threads.append(thread)

        # Start queue processor if not already running
        if not hasattr(self, '_queue_processor_running'):
            self._queue_processor_running = True
            self.process_response_queue()

        # Re-enable send button after a short delay (threads run in background)
        self.root.after(1000, lambda: self.send_button.configure(state="normal", text="Envoyer"))

    def format_structured_response(self, parsed_json):
        """Format a structured JSON response for display"""
        explanation = parsed_json.get("explanation", "")
        final_answer = parsed_json.get("final_answer", "")
        final_answer_letter = parsed_json.get("final_answer_letter", "")

        formatted = ""
        if explanation:
            formatted += "📋 Explication:\n"
            formatted += f"{explanation}\n\n"

        if final_answer:
            formatted += "✅ Réponse finale:\n"
            formatted += f"{final_answer}\n\n"

        if final_answer_letter:
            formatted += "🔤 Lettre de la réponse:\n"
            formatted += f"{final_answer_letter}"

        return formatted if formatted else json.dumps(parsed_json, indent=2, ensure_ascii=False)

    def build_context(self, query=""):
        """Build context from PDF knowledge base"""
        # If RAG is enabled and we have a query, use RAG
        if self.config.get("use_rag", False) and query:
            top_k = self.config.get("rag_top_k", 5)
            return self.rag_handler.build_rag_context(query, top_k=top_k)

        # Otherwise, use traditional full context
        if not self.pdf_knowledge:
            return ""

        context = "=== Base de Connaissances ===\n\n"
        for pdf in self.pdf_knowledge:
            context += f"Document: {pdf['filename']}\n"
            context += f"{pdf['content']}\n\n"
        context += "=== Fin de la Base de Connaissances ===\n\n"

        return context

    def add_message_to_chat(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.configure(state="normal")

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add separator if not first message
        if self.chat_display.get("1.0", "end-1c"):
            self.chat_display.insert("end", "\n" + "-" * 80 + "\n")

        # Add message
        self.chat_display.insert("end", f"[{timestamp}] {sender}:\n{message}\n")

        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def clear_chat(self):
        """Clear the chat display and history"""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.chat_history = []

        # Reset context
        self.last_sent_context = ""

        # Update context window if open
        self.update_context_window_if_open()

        # Reset context usage display
        self.context_label.configure(text="Utilisation du contexte: -- / --")
        self.context_progress.set(0)
        self.context_percentage.configure(text="0%")

    def update_context_usage(self, usage, model):
        """Update the context usage display"""
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)

        # Get context limit for the model from OpenRouter API
        context_limit = self.openrouter_client.get_context_limit(
            model,
            self.config.get("api_key")
        )

        # Calculate percentage
        percentage = (prompt_tokens / context_limit) * 100 if context_limit > 0 else 0

        # Update label
        self.context_label.configure(
            text=f"Utilisation du contexte: {prompt_tokens:,} / {context_limit:,} tokens"
        )

        # Update progress bar
        progress_value = min(prompt_tokens / context_limit, 1.0) if context_limit > 0 else 0
        self.context_progress.set(progress_value)

        # Update percentage label with color coding
        percentage_text = f"{percentage:.1f}%"
        self.context_percentage.configure(text=percentage_text)

        # Change color based on usage level
        if percentage < 50:
            color = "green"
        elif percentage < 80:
            color = "orange"
        else:
            color = "red"

        self.context_percentage.configure(text_color=color)

    def show_context_window(self):
        """Show a window with the complete context that will be sent to the LLM"""
        # Check if window already exists and is visible
        if self.context_window is not None and self.context_window.winfo_exists():
            # Window exists, just update content and focus
            self.update_context_window_content()
            self.context_window.focus()
            return

        # Create new toplevel window
        self.context_window = ctk.CTkToplevel(self.root)
        self.context_window.title("Contexte Complet du LLM")
        self.context_window.geometry("800x600")

        # Bind window close event to reset reference
        self.context_window.protocol("WM_DELETE_WINDOW", self.on_context_window_close)

        # Title
        title_label = ctk.CTkLabel(
            self.context_window,
            text="Contexte envoyé au LLM",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # Info label
        self.context_info_label = ctk.CTkLabel(
            self.context_window,
            text="",
            font=("Arial", 11),
            text_color="gray"
        )
        self.context_info_label.pack(pady=(0, 10))

        # Text display
        self.context_display = ctk.CTkTextbox(
            self.context_window,
            wrap="word",
            font=("Courier", 11)
        )
        self.context_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Update content
        self.update_context_window_content()

        # Close button
        close_button = ctk.CTkButton(
            self.context_window,
            text="Fermer",
            command=self.on_context_window_close,
            width=120,
            height=35
        )
        close_button.pack(pady=10)

        # Focus on the window
        self.context_window.focus()

    def on_context_window_close(self):
        """Handle context window close event"""
        if self.context_window is not None:
            self.context_window.destroy()
            self.context_window = None

    def update_context_window_if_open(self):
        """Update context window content if it's open"""
        if self.context_window is not None and self.context_window.winfo_exists():
            self.update_context_window_content()

    def update_context_window_content(self):
        """Update the content of the context window"""
        if self.context_window is None or not self.context_window.winfo_exists():
            return

        # Build complete context
        complete_context = self.build_complete_context()

        # Update info label
        mode = "RAG (recherche sémantique)" if self.config.get("use_rag", False) else "Traditionnel (contenu complet)"
        self.context_info_label.configure(
            text=f"Mode: {mode} | Mis à jour automatiquement lors de l'envoi de messages"
        )

        # Update text display
        self.context_display.configure(state="normal")
        self.context_display.delete("1.0", "end")
        self.context_display.insert("1.0", complete_context)
        self.context_display.configure(state="disabled")

    def build_complete_context(self):
        """Build the complete context that will be sent to the LLM"""
        context_parts = []

        # PDF Knowledge Base - Use last sent context if available
        if self.last_sent_context:
            context_parts.append("=" * 80)
            context_parts.append("CONTEXTE DES PDFs (DERNIER ENVOYÉ)")
            context_parts.append("=" * 80)
            context_parts.append(self.last_sent_context)
            context_parts.append("")
        elif self.pdf_knowledge:
            # No message sent yet, show preview
            context_parts.append("=" * 80)
            context_parts.append("CONTEXTE DES PDFs (PRÉVISUALISATION)")
            context_parts.append("=" * 80)
            if self.config.get("use_rag", False):
                context_parts.append("[Mode RAG activé - Le contexte sera généré lors de l'envoi d'un message]")
                context_parts.append("")
                stats = self.rag_handler.get_stats()
                context_parts.append(f"Statistiques RAG:")
                context_parts.append(f"  • Total de chunks indexés: {stats['total_chunks']}")
                context_parts.append(f"  • Nombre de PDFs: {stats['total_pdfs']}")
                context_parts.append("")
            else:
                pdf_context = self.build_context()
                context_parts.append(pdf_context)
            context_parts.append("")

        # Chat History
        if self.chat_history:
            context_parts.append("=" * 80)
            context_parts.append("HISTORIQUE DE CONVERSATION (10 DERNIERS MESSAGES)")
            context_parts.append("=" * 80)
            context_parts.append("")

            for i, msg in enumerate(self.chat_history[-10:], 1):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                context_parts.append(f"--- Message {i} ({role}) ---")
                context_parts.append(content)
                context_parts.append("")

        # If no context at all
        if not context_parts:
            context_parts.append("Aucun contexte disponible pour le moment.")
            context_parts.append("")
            context_parts.append("Le contexte sera créé après:")
            context_parts.append("• Avoir ajouté des PDFs dans l'onglet 'Base de Connaissances PDF'")
            context_parts.append("• Avoir envoyé des messages dans le chat")

        return "\n".join(context_parts)

    def toggle_rag(self):
        """Toggle RAG mode on/off"""
        is_enabled = self.rag_switch.get()
        self.config["use_rag"] = is_enabled
        self.save_config()

        # Update switch text
        self.rag_switch.configure(text="Activé" if is_enabled else "Désactivé")

        # Update info
        self.update_rag_info()

        # Show message
        if is_enabled:
            messagebox.showinfo(
                "RAG Activé",
                "Le mode RAG est maintenant activé.\n\n"
                "Les PDFs existants vont être indexés pour la recherche sémantique.\n"
                "Cela peut prendre quelques instants..."
            )
            # Process existing PDFs for RAG
            self.reindex_pdfs_for_rag()
        else:
            messagebox.showinfo(
                "RAG Désactivé",
                "Le mode RAG est désactivé.\n\n"
                "L'intégralité du contenu des PDFs sera envoyée au LLM."
            )

    def update_rag_info(self):
        """Update RAG information label"""
        stats = self.rag_handler.get_stats()
        if stats['total_chunks'] > 0:
            self.rag_info_label.configure(
                text=f"({stats['total_chunks']} chunks indexés)"
            )
        else:
            self.rag_info_label.configure(text="(Aucun chunk indexé)")

    def save_rag_parameters(self):
        """Save RAG parameters"""
        try:
            # Get and validate chunk size
            chunk_size = int(self.chunk_size_entry.get())
            if chunk_size < 50 or chunk_size > 2000:
                messagebox.showerror(
                    "Erreur",
                    "La taille des chunks doit être entre 50 et 2000 mots"
                )
                return

            # Get and validate chunk overlap
            chunk_overlap = int(self.chunk_overlap_entry.get())
            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                messagebox.showerror(
                    "Erreur",
                    "Le chevauchement doit être entre 0 et la taille des chunks"
                )
                return

            # Get and validate top k
            top_k = int(self.topk_entry.get())
            if top_k < 1 or top_k > 20:
                messagebox.showerror(
                    "Erreur",
                    "Le nombre de résultats doit être entre 1 et 20"
                )
                return

            # Save to config
            self.config["rag_chunk_size"] = chunk_size
            self.config["rag_chunk_overlap"] = chunk_overlap
            self.config["rag_top_k"] = top_k
            self.save_config()

            # Ask if user wants to reindex
            if self.config.get("use_rag", False) and self.pdf_knowledge:
                if messagebox.askyesno(
                    "Réindexation",
                    "Les paramètres ont été sauvegardés.\n\n"
                    "Voulez-vous réindexer les PDFs avec les nouveaux paramètres?\n"
                    "(Recommandé si vous avez modifié la taille des chunks ou le chevauchement)"
                ):
                    self.reindex_pdfs_for_rag()
            else:
                messagebox.showinfo(
                    "Succès",
                    "Les paramètres RAG ont été sauvegardés!"
                )

        except ValueError:
            messagebox.showerror(
                "Erreur",
                "Veuillez entrer des valeurs numériques valides"
            )

    def reindex_pdfs_for_rag(self):
        """Reindex all existing PDFs for RAG"""
        if not self.pdf_knowledge:
            return

        # Clear existing RAG data
        self.rag_handler.clear_all()

        # Get RAG parameters
        chunk_size = self.config.get("rag_chunk_size", 500)
        chunk_overlap = self.config.get("rag_chunk_overlap", 50)

        # Process each PDF
        for pdf in self.pdf_knowledge:
            try:
                chunks_count = self.rag_handler.process_pdf(
                    pdf['filename'],
                    pdf['content'],
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                print(f"Indexed {pdf['filename']}: {chunks_count} chunks")
            except Exception as e:
                print(f"Error indexing {pdf['filename']}: {e}")

        # Save embeddings
        self.rag_handler.save_embeddings()

        # Update info
        self.update_rag_info()

        messagebox.showinfo(
            "Indexation Terminée",
            f"Indexation terminée!\n\n"
            f"{self.rag_handler.get_stats()['total_chunks']} chunks créés pour "
            f"{len(self.pdf_knowledge)} PDF(s)."
        )

    def upload_pdf(self):
        """Upload and process a PDF file"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un PDF",
            filetypes=[("PDF files", "*.pdf")]
        )

        if not file_path:
            return

        try:
            # Extract text from PDF
            text = self.pdf_handler.extract_text(file_path)

            if not text.strip():
                messagebox.showwarning("Avertissement", "Le PDF semble vide ou le texte n'a pas pu être extrait.")
                return

            # Get filename
            filename = os.path.basename(file_path)

            # Check if already exists
            for pdf in self.pdf_knowledge:
                if pdf['filename'] == filename:
                    if not messagebox.askyesno("Confirmation", f"{filename} existe déjà. Voulez-vous le remplacer?"):
                        return
                    self.pdf_knowledge.remove(pdf)
                    # Remove from RAG if exists
                    self.rag_handler.remove_pdf(filename)
                    break

            # Add to knowledge base
            self.pdf_knowledge.append({
                "filename": filename,
                "content": text,
                "added_date": datetime.now().isoformat()
            })

            # Save to file
            self.save_pdf_knowledge()

            # Process for RAG if enabled
            if self.config.get("use_rag", False):
                try:
                    chunk_size = self.config.get("rag_chunk_size", 500)
                    chunk_overlap = self.config.get("rag_chunk_overlap", 50)
                    chunks_count = self.rag_handler.process_pdf(
                        filename,
                        text,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap
                    )
                    self.rag_handler.save_embeddings()
                    self.update_rag_info()
                    messagebox.showinfo(
                        "Succès",
                        f"{filename} a été ajouté à la base de connaissances!\n\n"
                        f"{chunks_count} chunks créés pour la recherche sémantique."
                    )
                except Exception as e:
                    messagebox.showwarning(
                        "Avertissement",
                        f"{filename} a été ajouté mais l'indexation RAG a échoué:\n{str(e)}\n\n"
                        "Le PDF sera disponible en mode traditionnel."
                    )
            else:
                messagebox.showinfo("Succès", f"{filename} a été ajouté à la base de connaissances!")

            # Update display
            self.update_pdf_list()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du traitement du PDF:\n{str(e)}")

    def update_pdf_list(self):
        """Update the PDF list display"""
        # Clear current list
        for widget in self.pdf_list_frame.winfo_children():
            widget.destroy()

        if not self.pdf_knowledge:
            label = ctk.CTkLabel(
                self.pdf_list_frame,
                text="Aucun PDF dans la base de connaissances",
                text_color="gray"
            )
            label.pack(pady=10)
            return

        # Add each PDF
        for i, pdf in enumerate(self.pdf_knowledge):
            pdf_frame = ctk.CTkFrame(self.pdf_list_frame)
            pdf_frame.pack(fill="x", padx=5, pady=5)

            # PDF info
            info_label = ctk.CTkLabel(
                pdf_frame,
                text=f"📄 {pdf['filename']}\n"
                     f"Ajouté: {pdf['added_date'][:10]}\n"
                     f"Taille: {len(pdf['content'])} caractères",
                anchor="w",
                justify="left"
            )
            info_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

            # Remove button
            remove_btn = ctk.CTkButton(
                pdf_frame,
                text="Supprimer",
                command=lambda idx=i: self.remove_pdf(idx),
                width=100,
                fg_color="red"
            )
            remove_btn.pack(side="right", padx=10, pady=5)

    def remove_pdf(self, index):
        """Remove a PDF from the knowledge base"""
        pdf = self.pdf_knowledge[index]
        if messagebox.askyesno("Confirmation", f"Voulez-vous vraiment supprimer {pdf['filename']}?"):
            # Remove from knowledge base
            self.pdf_knowledge.pop(index)
            self.save_pdf_knowledge()

            # Remove from RAG
            self.rag_handler.remove_pdf(pdf['filename'])
            self.rag_handler.save_embeddings()
            self.update_rag_info()

            # Update display
            self.update_pdf_list()

    def toggle_structured_output(self):
        """Toggle structured output on/off"""
        is_enabled = self.structured_output_switch.get()
        self.config["use_structured_output"] = is_enabled
        self.save_config()

        # Update switch text
        self.structured_output_switch.configure(text="Activé" if is_enabled else "Désactivé")

        # Show info message
        if is_enabled:
            messagebox.showinfo(
                "Structured Output Activé",
                "Les réponses du LLM seront structurées en JSON avec:\n\n"
                "• explanation: Explication détaillée\n"
                "• final_answer: Réponse finale\n"
                "• final_answer_letter: Lettre de la réponse (A, B, C, D...)\n\n"
                "Note: Cette fonctionnalité nécessite un modèle compatible\n"
                "(OpenAI GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)"
            )
        else:
            messagebox.showinfo(
                "Structured Output Désactivé",
                "Les réponses du LLM seront en texte libre."
            )

    def update_models_list(self):
        """Update the models list display"""
        # Clear existing widgets
        for widget in self.models_list_frame.winfo_children():
            widget.destroy()

        models = self.config.get("models", [])

        if not models:
            # No models
            no_models_label = ctk.CTkLabel(
                self.models_list_frame,
                text="Aucun modèle configuré\n\nAjoutez un modèle ci-dessous",
                font=("Arial", 11),
                text_color="gray"
            )
            no_models_label.pack(pady=20)
            return

        # Create a row for each model
        for i, model in enumerate(models):
            model_row = ctk.CTkFrame(self.models_list_frame)
            model_row.pack(fill="x", padx=5, pady=5)

            # Model name
            name_label = ctk.CTkLabel(
                model_row,
                text=f"🤖 {model['name']}",
                font=("Arial", 11),
                anchor="w"
            )
            name_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

            # Enable/Disable switch
            is_enabled = model.get("enabled", False)
            switch = ctk.CTkSwitch(
                model_row,
                text="Activé" if is_enabled else "Désactivé",
                command=lambda m=model['name']: self.toggle_model_enabled(m),
                font=("Arial", 10)
            )
            if is_enabled:
                switch.select()
            switch.pack(side="left", padx=5)

            # Remove button
            remove_btn = ctk.CTkButton(
                model_row,
                text="Supprimer",
                command=lambda m=model['name']: self.remove_model_from_list(m),
                width=90,
                height=28,
                fg_color="red"
            )
            remove_btn.pack(side="right", padx=5)

    def add_model_to_list(self):
        """Add a new model to the list"""
        model_name = self.new_model_entry.get().strip()

        if not model_name:
            messagebox.showerror("Erreur", "Veuillez entrer un nom de modèle")
            return

        # Add the model
        if self.add_model(model_name, enabled=True):
            # Clear input
            self.new_model_entry.delete(0, "end")

            # Update displays
            self.update_models_list()
            self.update_llm_cards()

            messagebox.showinfo("Succès", f"Modèle '{model_name}' ajouté avec succès!")
        else:
            messagebox.showerror("Erreur", f"Le modèle '{model_name}' existe déjà")

    def remove_model_from_list(self, model_name):
        """Remove a model from the list"""
        if messagebox.askyesno("Confirmation", f"Voulez-vous vraiment supprimer '{model_name}'?"):
            if self.remove_model(model_name):
                # Update displays
                self.update_models_list()
                self.update_llm_cards()

                messagebox.showinfo("Succès", f"Modèle '{model_name}' supprimé")
            else:
                messagebox.showerror("Erreur", "Impossible de supprimer le modèle")

    def toggle_model_enabled(self, model_name):
        """Toggle a model's enabled status"""
        if self.toggle_model(model_name):
            # Update displays
            self.update_models_list()
            self.update_llm_cards()

            # Get new status
            for model in self.config.get("models", []):
                if model["name"] == model_name:
                    status = "activé" if model.get("enabled", False) else "désactivé"
                    messagebox.showinfo("Succès", f"Modèle '{model_name}' {status}")
                    break

    def save_configuration(self):
        """Save the configuration"""
        api_key = self.api_key_entry.get().strip()

        if not api_key:
            messagebox.showerror("Erreur", "Veuillez entrer une clé API")
            return

        self.config["api_key"] = api_key
        self.save_config()

        self.config_status_label.configure(
            text="Configuration sauvegardée avec succès!",
            text_color="green"
        )

        # Clear status after 3 seconds
        self.root.after(3000, lambda: self.config_status_label.configure(text=""))

    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AILMApp()
    app.run()
