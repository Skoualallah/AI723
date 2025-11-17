import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime
import threading
from queue import Queue
import uuid
from pdf_handler import PDFHandler
from file_handler import FileHandler
from openrouter_client import OpenRouterClient
from google_client import GoogleClient
from rag_handler import RAGHandler

class AILMApp:
    def __init__(self):
        # Configuration
        self.config_file = "config.json"
        self.pdf_data_file = "pdf_knowledge_base.json"
        self.conversations_file = "conversations_history.json"
        self.config = self.load_config()
        self.pdf_knowledge = self.load_pdf_knowledge()

        # Initialize handlers
        self.pdf_handler = PDFHandler()
        self.file_handler = FileHandler()
        self.openrouter_client = OpenRouterClient()
        self.google_client = GoogleClient()
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

        # Store full responses for each model
        self.llm_responses = {}

        # Conversation management
        self.current_conversation_id = self.generate_conversation_id()
        self.current_conversation_messages = []
        self.conversations_history = self.load_conversations_history()

        # Histogram management (for real-time display)
        self.answer_counts = {}

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
        self.tab_history = self.tabview.add("Historique")
        self.tab_pdf = self.tabview.add("Base de Connaissances PDF")
        self.tab_config = self.tabview.add("Configuration")

        # Setup each tab
        self.setup_chat_tab()
        self.setup_llm_tab()
        self.setup_history_tab()
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

    def add_model(self, model_name, enabled=True, provider="openrouter"):
        """Add a new model to the list"""
        if "models" not in self.config:
            self.config["models"] = []

        # Check if model already exists
        for model in self.config["models"]:
            if model["name"] == model_name:
                return False  # Model already exists

        self.config["models"].append({
            "name": model_name,
            "enabled": enabled,
            "provider": provider  # "openrouter" or "google"
        })
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

    def generate_conversation_id(self):
        """Generate a unique conversation ID"""
        return str(uuid.uuid4())

    def load_conversations_history(self):
        """Load conversations history from file"""
        if os.path.exists(self.conversations_file):
            try:
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"conversations": []}

    def save_conversations_history(self):
        """Save conversations history to file"""
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations_history, f, indent=2, ensure_ascii=False)

    def save_current_message(self, user_message, context, responses, histogram):
        """Save the current message to the conversation history"""
        message_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": user_message,
            "context": context,
            "responses": responses,
            "histogram": histogram
        }
        self.current_conversation_messages.append(message_data)

        # Sauvegarder automatiquement après chaque message
        self.auto_save_conversation()

    def save_conversation(self):
        """Save the current conversation to history"""
        if not self.current_conversation_messages:
            return

        conversation = {
            "id": self.current_conversation_id,
            "start_timestamp": self.current_conversation_messages[0]["timestamp"],
            "messages": self.current_conversation_messages
        }

        # Add to history
        self.conversations_history["conversations"].insert(0, conversation)
        self.save_conversations_history()

    def auto_save_conversation(self):
        """Automatically save/update the current conversation after each message"""
        if not self.current_conversation_messages:
            return

        # Look for existing conversation with this ID
        existing_index = None
        for i, conv in enumerate(self.conversations_history["conversations"]):
            if conv["id"] == self.current_conversation_id:
                existing_index = i
                break

        conversation = {
            "id": self.current_conversation_id,
            "start_timestamp": self.current_conversation_messages[0]["timestamp"],
            "messages": self.current_conversation_messages
        }

        if existing_index is not None:
            # Update existing conversation
            self.conversations_history["conversations"][existing_index] = conversation
        else:
            # Create new conversation at the beginning
            self.conversations_history["conversations"].insert(0, conversation)

        # Save to file and update UI
        self.save_conversations_history()
        self.update_history_list()

    def new_conversation(self):
        """Start a new conversation"""
        # Current conversation is already auto-saved, no need to save again
        # (it's saved automatically after each message via auto_save_conversation)

        # Reset for new conversation
        self.current_conversation_id = self.generate_conversation_id()
        self.current_conversation_messages = []
        self.chat_history = []
        self.answer_counts = {}

        # Clear chat display
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

        # Reset context
        self.last_sent_context = ""
        self.update_context_window_if_open()

        # Reset histogram
        self.reset_histogram()

        # Reset LLM statuses to idle and clear responses
        for model_name in self.get_enabled_models():
            if model_name in self.llm_status_widgets:
                self.update_llm_status(model_name, "idle", {"answer_letter": "?"})
            if model_name in self.llm_responses:
                self.llm_responses[model_name] = {
                    "content": "",
                    "error": "",
                    "answer_letter": "?"
                }

        # Update history tab
        self.update_history_list()

    def setup_chat_tab(self):
        """Setup the chat interface tab"""
        # Answer histogram frame (for structured output)
        self.histogram_frame = ctk.CTkFrame(self.tab_chat)
        self.histogram_frame.pack(fill="x", padx=10, pady=(10, 0))

        histogram_title = ctk.CTkLabel(
            self.histogram_frame,
            text="📊 Distribution des Réponses",
            font=("Arial", 12, "bold")
        )
        histogram_title.pack(side="left", padx=10, pady=5)

        # Histogram bars container
        self.histogram_bars_frame = ctk.CTkFrame(self.histogram_frame, fg_color="transparent")
        self.histogram_bars_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)

        # Dictionary to store answer counts and label widgets
        self.answer_counts = {}
        self.histogram_labels = {}

        # Hide histogram by default (shown when structured output is active)
        self.histogram_frame.pack_forget()

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

        # New conversation button
        self.new_conv_button = ctk.CTkButton(
            button_frame,
            text="Nouvelle\nConversation",
            command=self.new_conversation,
            width=100,
            height=35,
            fg_color="#2B7A78"
        )
        self.new_conv_button.pack()

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
        """Create a compact monitoring row for a specific LLM"""
        # Row frame - clickable
        row_frame = ctk.CTkFrame(self.llm_cards_frame, cursor="hand2")
        row_frame.pack(fill="x", padx=5, pady=3)

        # Make the frame clickable
        row_frame.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Status indicator (emoji)
        status_label = ctk.CTkLabel(
            row_frame,
            text="⚪",
            font=("Arial", 16),
            width=30
        )
        status_label.pack(side="left", padx=(10, 5))
        status_label.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Model name
        model_label = ctk.CTkLabel(
            row_frame,
            text=model_name,
            font=("Arial", 11, "bold"),
            anchor="w",
            width=250
        )
        model_label.pack(side="left", padx=5)
        model_label.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Answer letter
        answer_label = ctk.CTkLabel(
            row_frame,
            text="?",
            font=("Arial", 18, "bold"),
            text_color="gray",
            width=40
        )
        answer_label.pack(side="left", padx=5)
        answer_label.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Context usage compact
        context_label = ctk.CTkLabel(
            row_frame,
            text="--",
            font=("Arial", 10),
            width=60
        )
        context_label.pack(side="left", padx=5)
        context_label.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Mini progress bar
        context_progress = ctk.CTkProgressBar(
            row_frame,
            width=80,
            height=8
        )
        context_progress.pack(side="left", padx=5)
        context_progress.set(0)
        context_progress.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Timestamp
        timestamp_label = ctk.CTkLabel(
            row_frame,
            text="",
            font=("Arial", 9),
            text_color="gray",
            anchor="e",
            width=80
        )
        timestamp_label.pack(side="right", padx=10)
        timestamp_label.bind("<Button-1>", lambda e: self.show_llm_response(model_name))

        # Initialize response storage
        self.llm_responses[model_name] = {
            "content": "",
            "error": "",
            "answer_letter": "?"
        }

        # Store widgets for this model
        self.llm_status_widgets[model_name] = {
            "row_frame": row_frame,
            "status_label": status_label,
            "model_label": model_label,
            "answer_label": answer_label,
            "context_label": context_label,
            "context_progress": context_progress,
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
                - answer_letter: Letter of the answer
        """
        if model_name not in self.llm_status_widgets:
            return

        widgets = self.llm_status_widgets[model_name]
        data = data or {}

        # Update status indicator (emoji only)
        status_emojis = {
            "idle": "⚪",
            "processing": "🟡",
            "completed": "🟢",
            "error": "🔴"
        }

        emoji = status_emojis.get(status, "⚪")
        widgets["status_label"].configure(text=emoji)

        # Update context usage if available
        if "usage" in data:
            usage = data["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            context_limit = data.get("context_limit", 8192)
            percentage = (prompt_tokens / context_limit) * 100 if context_limit > 0 else 0

            widgets["context_label"].configure(text=f"{percentage:.0f}%")
            widgets["context_progress"].set(min(prompt_tokens / context_limit, 1.0) if context_limit > 0 else 0)

        # Store response/error for popup display
        if "error" in data:
            self.llm_responses[model_name]["error"] = data["error"]
            self.llm_responses[model_name]["content"] = ""
        elif "response" in data:
            self.llm_responses[model_name]["content"] = data["response"]
            self.llm_responses[model_name]["error"] = ""

        # Update answer letter
        if "answer_letter" in data:
            answer_letter = data["answer_letter"]
            self.llm_responses[model_name]["answer_letter"] = answer_letter

            # Color based on status
            if status == "error":
                color = "red"
            elif status == "completed":
                color = "green"
            else:
                color = "gray"

            widgets["answer_label"].configure(text=answer_letter, text_color=color)

        # Update timestamp
        if "timestamp" in data:
            widgets["timestamp_label"].configure(text=data["timestamp"])

    def show_llm_response(self, model_name):
        """Show full response from a specific LLM in a popup window"""
        if model_name not in self.llm_responses:
            return

        response_data = self.llm_responses[model_name]

        # Create popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title(f"Réponse - {model_name}")
        popup.geometry("700x500")

        # Header with model name and answer letter
        header_frame = ctk.CTkFrame(popup)
        header_frame.pack(fill="x", padx=10, pady=10)

        title_label = ctk.CTkLabel(
            header_frame,
            text=f"🤖 {model_name}",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side="left", padx=10)

        answer_letter_label = ctk.CTkLabel(
            header_frame,
            text=f"Réponse: {response_data.get('answer_letter', '?')}",
            font=("Arial", 18, "bold"),
            text_color="green"
        )
        answer_letter_label.pack(side="right", padx=10)

        # Content area
        content_frame = ctk.CTkFrame(popup)
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Display content or error
        if response_data.get("error"):
            # Show error
            error_label = ctk.CTkLabel(
                content_frame,
                text="❌ Erreur",
                font=("Arial", 14, "bold"),
                text_color="red"
            )
            error_label.pack(pady=10)

            error_text = ctk.CTkTextbox(
                content_frame,
                wrap="word",
                font=("Arial", 11)
            )
            error_text.pack(fill="both", expand=True, padx=10, pady=10)
            error_text.insert("1.0", response_data["error"])
            error_text.configure(state="disabled")

        elif response_data.get("content"):
            # Show response
            response_text = ctk.CTkTextbox(
                content_frame,
                wrap="word",
                font=("Arial", 11)
            )
            response_text.pack(fill="both", expand=True, padx=10, pady=10)

            # Format if structured output
            if self.config.get("use_structured_output", False):
                try:
                    parsed = json.loads(response_data["content"])
                    formatted = self.format_structured_response(parsed)
                    response_text.insert("1.0", formatted)
                except json.JSONDecodeError:
                    response_text.insert("1.0", response_data["content"])
            else:
                response_text.insert("1.0", response_data["content"])

            response_text.configure(state="disabled")

        else:
            # No response yet
            no_response_label = ctk.CTkLabel(
                content_frame,
                text="Aucune réponse disponible\n\nEn attente de la réponse du modèle...",
                font=("Arial", 12),
                text_color="gray"
            )
            no_response_label.pack(expand=True)

        # Close button
        close_btn = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy,
            width=120,
            height=35
        )
        close_btn.pack(pady=10)

        # Focus on the window
        popup.focus()

    def setup_history_tab(self):
        """Setup the history tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_history,
            text="Historique des Conversations",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        # Controls frame
        controls_frame = ctk.CTkFrame(self.tab_history, fg_color="transparent")
        controls_frame.pack(fill="x", padx=20, pady=(0, 10))

        # Delete all button
        delete_all_btn = ctk.CTkButton(
            controls_frame,
            text="Tout Supprimer",
            command=self.delete_all_conversations,
            width=150,
            height=35,
            fg_color="red"
        )
        delete_all_btn.pack(side="right", padx=5)

        # Conversations list
        self.history_list_frame = ctk.CTkScrollableFrame(
            self.tab_history,
            label_text="Conversations"
        )
        self.history_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Initialize history list
        self.update_history_list()

    def update_history_list(self):
        """Update the conversations history list"""
        # Clear existing widgets
        for widget in self.history_list_frame.winfo_children():
            widget.destroy()

        conversations = self.conversations_history.get("conversations", [])

        if not conversations:
            # No conversations
            no_conv_label = ctk.CTkLabel(
                self.history_list_frame,
                text="Aucune conversation enregistrée\n\nCommencez à discuter dans l'onglet Chat",
                font=("Arial", 12),
                text_color="gray"
            )
            no_conv_label.pack(pady=50)
            return

        # Create a row for each conversation
        for conv in conversations:
            conv_frame = ctk.CTkFrame(self.history_list_frame, cursor="hand2")
            conv_frame.pack(fill="x", padx=5, pady=5)

            # Make clickable
            conv_frame.bind("<Button-1>", lambda e, c=conv: self.show_conversation_detail(c))

            # Info frame
            info_frame = ctk.CTkFrame(conv_frame, fg_color="transparent")
            info_frame.pack(fill="x", padx=10, pady=5)

            # Timestamp
            timestamp_label = ctk.CTkLabel(
                info_frame,
                text=f"📅 {conv['start_timestamp']}",
                font=("Arial", 11, "bold"),
                anchor="w"
            )
            timestamp_label.pack(side="left", padx=5)
            timestamp_label.bind("<Button-1>", lambda e, c=conv: self.show_conversation_detail(c))

            # Message count
            msg_count = len(conv.get("messages", []))
            count_label = ctk.CTkLabel(
                info_frame,
                text=f"{msg_count} message{'s' if msg_count > 1 else ''}",
                font=("Arial", 10),
                text_color="gray"
            )
            count_label.pack(side="left", padx=10)
            count_label.bind("<Button-1>", lambda e, c=conv: self.show_conversation_detail(c))

            # First message preview
            if conv.get("messages"):
                first_msg = conv["messages"][0]["user_message"]
                preview = first_msg[:80] + "..." if len(first_msg) > 80 else first_msg

                preview_label = ctk.CTkLabel(
                    conv_frame,
                    text=preview,
                    font=("Arial", 10),
                    text_color="lightgray",
                    anchor="w"
                )
                preview_label.pack(fill="x", padx=15, pady=(0, 5))
                preview_label.bind("<Button-1>", lambda e, c=conv: self.show_conversation_detail(c))

    def delete_all_conversations(self):
        """Delete all conversations from history"""
        if messagebox.askyesno("Confirmation", "Voulez-vous vraiment supprimer tout l'historique?"):
            self.conversations_history = {"conversations": []}
            self.save_conversations_history()
            self.update_history_list()
            messagebox.showinfo("Succès", "Tout l'historique a été supprimé")

    def show_conversation_detail(self, conversation):
        """Show detailed view of a conversation in a popup"""
        # Create popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title(f"Conversation - {conversation['start_timestamp']}")
        popup.geometry("900x700")

        # Header
        header_frame = ctk.CTkFrame(popup)
        header_frame.pack(fill="x", padx=10, pady=10)

        title_label = ctk.CTkLabel(
            header_frame,
            text=f"💬 Conversation du {conversation['start_timestamp']}",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side="left", padx=10)

        # Scrollable content
        content_frame = ctk.CTkScrollableFrame(popup)
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Display each message
        for i, msg in enumerate(conversation.get("messages", [])):
            # Message container
            msg_frame = ctk.CTkFrame(content_frame)
            msg_frame.pack(fill="x", padx=5, pady=10)

            # Message header
            msg_header = ctk.CTkFrame(msg_frame, fg_color="transparent")
            msg_header.pack(fill="x", padx=10, pady=5)

            msg_time = ctk.CTkLabel(
                msg_header,
                text=f"⏱️ {msg['timestamp']}",
                font=("Arial", 11, "bold")
            )
            msg_time.pack(side="left", padx=5)

            # View context button
            context_btn = ctk.CTkButton(
                msg_header,
                text="Voir Contexte",
                command=lambda ctx=msg['context']: self.show_context_popup(ctx),
                width=120,
                height=25,
                fg_color="#2B5278"
            )
            context_btn.pack(side="right", padx=5)

            # User question
            question_label = ctk.CTkLabel(
                msg_frame,
                text="❓ Question:",
                font=("Arial", 11, "bold")
            )
            question_label.pack(anchor="w", padx=10, pady=(5, 0))

            question_text = ctk.CTkTextbox(
                msg_frame,
                height=60,
                font=("Arial", 11)
            )
            question_text.pack(fill="x", padx=10, pady=5)
            question_text.insert("1.0", msg['user_message'])
            question_text.configure(state="disabled")

            # Histogram if available
            if msg.get("histogram"):
                hist_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
                hist_frame.pack(fill="x", padx=10, pady=5)

                hist_label = ctk.CTkLabel(
                    hist_frame,
                    text="📊 Distribution globale:",
                    font=("Arial", 11, "bold")
                )
                hist_label.pack(side="left", padx=5)

                for letter, count in sorted(msg["histogram"].items()):
                    letter_count = ctk.CTkLabel(
                        hist_frame,
                        text=f"{letter}: {count}",
                        font=("Arial", 12, "bold"),
                        text_color="green"
                    )
                    letter_count.pack(side="left", padx=10)

            # Responses
            responses_label = ctk.CTkLabel(
                msg_frame,
                text="🤖 Réponses des LLMs:",
                font=("Arial", 11, "bold")
            )
            responses_label.pack(anchor="w", padx=10, pady=(10, 5))

            for resp in msg.get("responses", []):
                # Response row
                resp_row = ctk.CTkFrame(msg_frame, cursor="hand2")
                resp_row.pack(fill="x", padx=15, pady=3)

                # Make clickable to view full response
                resp_row.bind("<Button-1>", lambda e, r=resp: self.show_response_popup(r))

                # Model name
                model_label = ctk.CTkLabel(
                    resp_row,
                    text=f"• {resp['model']}",
                    font=("Arial", 10, "bold"),
                    anchor="w",
                    width=300
                )
                model_label.pack(side="left", padx=5)
                model_label.bind("<Button-1>", lambda e, r=resp: self.show_response_popup(r))

                # Answer letter
                if resp.get("answer_letter"):
                    answer_label = ctk.CTkLabel(
                        resp_row,
                        text=resp["answer_letter"],
                        font=("Arial", 14, "bold"),
                        text_color="green" if not resp.get("error") else "red",
                        width=30
                    )
                    answer_label.pack(side="left", padx=5)
                    answer_label.bind("<Button-1>", lambda e, r=resp: self.show_response_popup(r))

                # Status
                if resp.get("error"):
                    status_label = ctk.CTkLabel(
                        resp_row,
                        text="❌ Erreur",
                        font=("Arial", 9),
                        text_color="red"
                    )
                    status_label.pack(side="left", padx=5)
                    status_label.bind("<Button-1>", lambda e, r=resp: self.show_response_popup(r))

            # Separator between messages
            if i < len(conversation["messages"]) - 1:
                separator = ctk.CTkLabel(
                    content_frame,
                    text="─" * 80,
                    font=("Arial", 8),
                    text_color="gray"
                )
                separator.pack(pady=5)

        # Close button
        close_btn = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy,
            width=120,
            height=35
        )
        close_btn.pack(pady=10)

        popup.focus()

    def show_context_popup(self, context):
        """Show context in a popup"""
        popup = ctk.CTkToplevel(self.root)
        popup.title("Contexte Envoyé")
        popup.geometry("700x500")

        title_label = ctk.CTkLabel(
            popup,
            text="📄 Contexte Envoyé au LLM",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        context_text = ctk.CTkTextbox(
            popup,
            wrap="word",
            font=("Courier", 10)
        )
        context_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        context_text.insert("1.0", context if context else "Aucun contexte")
        context_text.configure(state="disabled")

        close_btn = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy,
            width=120,
            height=35
        )
        close_btn.pack(pady=10)

        popup.focus()

    def show_response_popup(self, response):
        """Show full response in a popup"""
        popup = ctk.CTkToplevel(self.root)
        popup.title(f"Réponse - {response['model']}")
        popup.geometry("700x500")

        # Header
        header_frame = ctk.CTkFrame(popup)
        header_frame.pack(fill="x", padx=10, pady=10)

        title_label = ctk.CTkLabel(
            header_frame,
            text=f"🤖 {response['model']}",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side="left", padx=10)

        if response.get("answer_letter"):
            answer_label = ctk.CTkLabel(
                header_frame,
                text=f"Réponse: {response['answer_letter']}",
                font=("Arial", 18, "bold"),
                text_color="green" if not response.get("error") else "red"
            )
            answer_label.pack(side="right", padx=10)

        # Content
        content_frame = ctk.CTkFrame(popup)
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if response.get("error"):
            error_label = ctk.CTkLabel(
                content_frame,
                text="❌ Erreur",
                font=("Arial", 14, "bold"),
                text_color="red"
            )
            error_label.pack(pady=10)

            error_text = ctk.CTkTextbox(
                content_frame,
                wrap="word",
                font=("Arial", 11)
            )
            error_text.pack(fill="both", expand=True, padx=10, pady=10)
            error_text.insert("1.0", response["error"])
            error_text.configure(state="disabled")
        else:
            response_text = ctk.CTkTextbox(
                content_frame,
                wrap="word",
                font=("Arial", 11)
            )
            response_text.pack(fill="both", expand=True, padx=10, pady=10)
            response_text.insert("1.0", response.get("content", "Aucune réponse"))
            response_text.configure(state="disabled")

        close_btn = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy,
            width=120,
            height=35
        )
        close_btn.pack(pady=10)

        popup.focus()

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
            text="📁 Ajouter un Fichier",
            command=self.upload_document,
            width=200,
            height=40
        )
        upload_btn.pack(pady=10)

        # Supported formats info
        formats_label = ctk.CTkLabel(
            self.tab_pdf,
            text="Formats supportés: TXT, MD, PDF, Python, JavaScript, JSON, CSV, XML, HTML, CSS, et plus...",
            font=("Arial", 10),
            text_color="gray"
        )
        formats_label.pack(pady=(0, 5))

        # Document list frame
        list_frame = ctk.CTkFrame(self.tab_pdf)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # List label
        list_label = ctk.CTkLabel(
            list_frame,
            text="Documents dans la base de connaissances:",
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

        # API Keys section
        api_frame = ctk.CTkFrame(self.tab_config)
        api_frame.pack(fill="x", padx=20, pady=10)

        # OpenRouter API Key
        api_label = ctk.CTkLabel(
            api_frame,
            text="🔑 Clé API OpenRouter:",
            font=("Arial", 14, "bold")
        )
        api_label.pack(pady=(5, 0), anchor="w", padx=10)

        self.api_key_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="Entrez votre clé API OpenRouter",
            width=400,
            show="*"
        )
        self.api_key_entry.pack(pady=5)
        if self.config.get("api_key"):
            self.api_key_entry.insert(0, self.config["api_key"])

        # Google AI Studio API Key
        google_api_label = ctk.CTkLabel(
            api_frame,
            text="🔑 Clé API Google AI Studio (Gemini):",
            font=("Arial", 14, "bold")
        )
        google_api_label.pack(pady=(15, 0), anchor="w", padx=10)

        self.google_api_key_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="Entrez votre clé API Google AI Studio",
            width=400,
            show="*"
        )
        self.google_api_key_entry.pack(pady=5)
        if self.config.get("google_api_key"):
            self.google_api_key_entry.insert(0, self.config["google_api_key"])

        # API info
        api_info_label = ctk.CTkLabel(
            api_frame,
            text="💡 OpenRouter pour Claude, GPT, etc. | Google AI Studio pour les modèles Gemini",
            font=("Arial", 10),
            text_color="gray"
        )
        api_info_label.pack(pady=(5, 10))

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

        # Provider dropdown
        self.provider_dropdown = ctk.CTkComboBox(
            add_model_frame,
            values=["OpenRouter", "Google AI"],
            width=130,
            state="readonly"
        )
        self.provider_dropdown.set("OpenRouter")
        self.provider_dropdown.pack(side="left", padx=5)

        self.new_model_entry = ctk.CTkEntry(
            add_model_frame,
            placeholder_text="Ex: anthropic/claude-3.5-sonnet",
            width=250
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
            text="OpenRouter: anthropic/claude-3.5-sonnet, openai/gpt-4-turbo | Google AI: gemini-1.5-pro, gemini-1.5-flash",
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

    def llm_worker(self, model_name, message, context, api_key, use_structured_output, provider="openrouter", google_api_key=None):
        """
        Worker thread function to send a request to a single LLM

        Args:
            model_name: Name of the model
            message: User message
            context: Context from RAG/PDFs
            api_key: OpenRouter API key
            use_structured_output: Whether to use structured output
            provider: Provider type ("openrouter" or "google")
            google_api_key: Google AI Studio API key (if provider is "google")
        """
        try:
            # Update status to processing
            self.response_queue.put({
                "type": "status",
                "model": model_name,
                "status": "processing",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Send request based on provider
            if provider == "google":
                # Use Google AI client
                if not google_api_key:
                    raise Exception("Clé API Google manquante")

                # Configure Google client if not already configured
                if not self.google_client.configured:
                    self.google_client.configure(google_api_key)

                # Prepare system prompt with context
                system_prompt = None
                if context:
                    system_prompt = f"Contexte:\n{context}"

                result = self.google_client.send_message(
                    model_name,
                    message,
                    system_prompt=system_prompt,
                    use_structured_output=use_structured_output
                )

                context_limit = result.get('context_length', 32768)
            else:
                # Use OpenRouter client
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
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "provider": provider
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
                    # Extract answer letter if structured output
                    answer_letter = "?"
                    if self.config.get("use_structured_output", False):
                        try:
                            parsed = json.loads(response["content"])
                            answer_letter = parsed.get("final_answer_letter", "?").strip().upper()
                        except json.JSONDecodeError:
                            pass

                    # Update LLM status with response
                    self.update_llm_status(
                        response["model"],
                        response["status"],
                        {
                            "response": response["content"],
                            "usage": response["usage"],
                            "context_limit": response["context_limit"],
                            "timestamp": response["timestamp"],
                            "answer_letter": answer_letter
                        }
                    )

                    # Store response for history
                    if hasattr(self, 'current_message_data'):
                        self.current_message_data["responses"][response["model"]] = {
                            "model": response["model"],
                            "content": response["content"],
                            "answer_letter": answer_letter,
                            "error": None
                        }

                        # Update histogram for this specific message
                        if answer_letter and answer_letter != "?":
                            if answer_letter not in self.current_message_data["histogram"]:
                                self.current_message_data["histogram"][answer_letter] = 0
                            self.current_message_data["histogram"][answer_letter] += 1

                        # Check if all responses received
                        if len(self.current_message_data["responses"]) == self.current_message_data["expected_count"]:
                            # Save message to history with its specific histogram
                            self.save_current_message(
                                self.current_message_data["user_message"],
                                self.current_message_data["context"],
                                list(self.current_message_data["responses"].values()),
                                dict(self.current_message_data["histogram"])
                            )
                            # Update history list
                            self.update_history_list()

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
                            "timestamp": response["timestamp"],
                            "answer_letter": "?"
                        }
                    )

                    # Store error for history
                    if hasattr(self, 'current_message_data'):
                        self.current_message_data["responses"][response["model"]] = {
                            "model": response["model"],
                            "content": "",
                            "answer_letter": "?",
                            "error": response["error"]
                        }

                        # Check if all responses received (including errors)
                        if len(self.current_message_data["responses"]) == self.current_message_data["expected_count"]:
                            # Save message to history with its specific histogram
                            self.save_current_message(
                                self.current_message_data["user_message"],
                                self.current_message_data["context"],
                                list(self.current_message_data["responses"].values()),
                                dict(self.current_message_data["histogram"])
                            )
                            # Update history list
                            self.update_history_list()

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

                # Update histogram if final_answer_letter is present
                answer_letter = parsed_response.get("final_answer_letter", "").strip().upper()
                if answer_letter:
                    self.update_histogram(answer_letter)

            except json.JSONDecodeError:
                display_content = content
        else:
            display_content = content

        # Add to chat with model name
        self.add_message_to_chat(f"Assistant ({model_name})", display_content)

    def reset_histogram(self):
        """Reset the histogram for a new question"""
        self.answer_counts = {}

        # Clear histogram labels
        for widget in self.histogram_bars_frame.winfo_children():
            widget.destroy()
        self.histogram_labels = {}

        # Show or hide histogram based on structured output setting
        if self.config.get("use_structured_output", False):
            self.histogram_frame.pack(fill="x", padx=10, pady=(10, 0), before=self.chat_display)
        else:
            self.histogram_frame.pack_forget()

    def update_histogram(self, answer_letter):
        """Update the histogram with a new answer letter"""
        if not answer_letter:
            return

        # Increment count
        if answer_letter not in self.answer_counts:
            self.answer_counts[answer_letter] = 0
        self.answer_counts[answer_letter] += 1

        # Update display
        self.render_histogram()

    def render_histogram(self):
        """Render the histogram visualization"""
        # Clear existing widgets
        for widget in self.histogram_bars_frame.winfo_children():
            widget.destroy()
        self.histogram_labels = {}

        if not self.answer_counts:
            return

        # Sort by letter
        sorted_letters = sorted(self.answer_counts.keys())

        # Create a label for each letter
        for letter in sorted_letters:
            count = self.answer_counts[letter]

            # Container for each bar
            bar_container = ctk.CTkFrame(self.histogram_bars_frame, fg_color="transparent")
            bar_container.pack(side="left", padx=5)

            # Letter label
            letter_label = ctk.CTkLabel(
                bar_container,
                text=letter,
                font=("Arial", 14, "bold"),
                width=30
            )
            letter_label.pack()

            # Count label with visual emphasis
            count_label = ctk.CTkLabel(
                bar_container,
                text=str(count),
                font=("Arial", 16, "bold"),
                text_color="green",
                width=30
            )
            count_label.pack()

            # Visual bar representation (progress bar)
            max_count = max(self.answer_counts.values())
            bar_height = min((count / max_count) * 50, 50) if max_count > 0 else 0

            bar = ctk.CTkProgressBar(
                bar_container,
                width=30,
                height=int(bar_height),
                orientation="vertical"
            )
            bar.set(1.0)  # Full bar, height represents the value
            bar.pack(pady=2)

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

        # Reset histogram for new question
        self.reset_histogram()

        # Prepare to collect responses for this message
        self.current_message_data = {
            "user_message": message,
            "context": context,
            "responses": {},
            "histogram": {},  # Histogramme spécifique à cette question
            "expected_count": len(enabled_models)
        }

        # Reset LLM statuses to idle
        for model_name in enabled_models:
            self.update_llm_status(model_name, "idle", {})

        # Start worker threads for each enabled model
        self.active_threads = []
        for model_data in self.config.get("models", []):
            if not model_data.get("enabled", False):
                continue

            model_name = model_data["name"]
            provider = model_data.get("provider", "openrouter")

            thread = threading.Thread(
                target=self.llm_worker,
                args=(
                    model_name,
                    message,
                    context,
                    self.config.get("api_key", ""),
                    self.config.get("use_structured_output", False),
                    provider,
                    self.config.get("google_api_key", "")
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

    def upload_document(self):
        """Upload and process a document file (supports multiple formats)"""
        # Get supported file types from FileHandler
        filetypes = self.file_handler.get_filetypes_for_dialog()
        # Add PDF support
        filetypes.insert(1, ("PDF", "*.pdf"))

        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier",
            filetypes=filetypes
        )

        if not file_path:
            return

        try:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()

            # Determine if it's a PDF or other file type
            if ext == '.pdf':
                # Use PDF handler for PDFs
                text = self.pdf_handler.extract_text(file_path)
                file_type_info = {
                    'type': 'pdf',
                    'emoji': '📕',
                    'name': 'PDF',
                    'extension': '.pdf'
                }
            else:
                # Use file handler for other types
                file_type_info = self.file_handler.get_file_type(filename)
                if not file_type_info:
                    messagebox.showerror("Erreur", f"Type de fichier non supporté: {ext}")
                    return
                text = self.file_handler.extract_text(file_path)

            if not text.strip():
                messagebox.showwarning("Avertissement", "Le fichier semble vide ou le texte n'a pas pu être extrait.")
                return

            # Check if already exists
            for doc in self.pdf_knowledge:
                if doc['filename'] == filename:
                    if not messagebox.askyesno("Confirmation", f"{filename} existe déjà. Voulez-vous le remplacer?"):
                        return
                    self.pdf_knowledge.remove(doc)
                    # Remove from RAG if exists
                    self.rag_handler.remove_pdf(filename)
                    break

            # Add to knowledge base with file type info
            self.pdf_knowledge.append({
                "filename": filename,
                "content": text,
                "added_date": datetime.now().isoformat(),
                "file_type": file_type_info
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
                        f"{file_type_info['emoji']} {filename} a été ajouté à la base de connaissances!\n\n"
                        f"{chunks_count} chunks créés pour la recherche sémantique."
                    )
                except Exception as e:
                    messagebox.showwarning(
                        "Avertissement",
                        f"{file_type_info['emoji']} {filename} a été ajouté mais l'indexation RAG a échoué:\n{str(e)}\n\n"
                        "Le document sera disponible en mode traditionnel."
                    )
            else:
                messagebox.showinfo("Succès", f"{file_type_info['emoji']} {filename} a été ajouté à la base de connaissances!")

            # Update display
            self.update_pdf_list()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du traitement du fichier:\n{str(e)}")

    def update_pdf_list(self):
        """Update the document list display"""
        # Clear current list
        for widget in self.pdf_list_frame.winfo_children():
            widget.destroy()

        if not self.pdf_knowledge:
            label = ctk.CTkLabel(
                self.pdf_list_frame,
                text="Aucun document dans la base de connaissances",
                text_color="gray"
            )
            label.pack(pady=10)
            return

        # Add each document
        for i, doc in enumerate(self.pdf_knowledge):
            doc_frame = ctk.CTkFrame(self.pdf_list_frame)
            doc_frame.pack(fill="x", padx=5, pady=5)

            # Get file type info (for backward compatibility with old PDFs)
            if 'file_type' in doc and doc['file_type']:
                file_emoji = doc['file_type']['emoji']
                file_type_name = doc['file_type']['name']
            else:
                # Default for old PDFs without file_type info
                file_emoji = "📕"
                file_type_name = "PDF"

            # Document info
            info_label = ctk.CTkLabel(
                doc_frame,
                text=f"{file_emoji} {doc['filename']} ({file_type_name})\n"
                     f"Ajouté: {doc['added_date'][:10]}\n"
                     f"Taille: {len(doc['content'])} caractères",
                anchor="w",
                justify="left"
            )
            info_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

            # Remove button
            remove_btn = ctk.CTkButton(
                doc_frame,
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

            # Get provider (default to openrouter for backward compatibility)
            provider = model.get("provider", "openrouter")
            provider_icon = "🔷" if provider == "google" else "🔶"
            provider_name = "Google AI" if provider == "google" else "OpenRouter"

            # Model name with provider badge
            name_label = ctk.CTkLabel(
                model_row,
                text=f"{provider_icon} {model['name']} ({provider_name})",
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

        # Get selected provider
        provider_display = self.provider_dropdown.get()
        provider = "google" if provider_display == "Google AI" else "openrouter"

        # Add the model
        if self.add_model(model_name, enabled=True, provider=provider):
            # Clear input
            self.new_model_entry.delete(0, "end")

            # Update displays
            self.update_models_list()
            self.update_llm_cards()

            provider_name = "Google AI" if provider == "google" else "OpenRouter"
            messagebox.showinfo("Succès", f"Modèle '{model_name}' ({provider_name}) ajouté avec succès!")
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
        google_api_key = self.google_api_key_entry.get().strip()

        # At least one API key must be provided
        if not api_key and not google_api_key:
            messagebox.showerror("Erreur", "Veuillez entrer au moins une clé API (OpenRouter ou Google)")
            return

        # Save API keys
        self.config["api_key"] = api_key
        self.config["google_api_key"] = google_api_key

        # Configure clients if keys are provided
        if google_api_key:
            try:
                self.google_client.configure(google_api_key)
            except Exception as e:
                messagebox.showwarning("Avertissement", f"Erreur lors de la configuration de Google AI: {str(e)}")

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
