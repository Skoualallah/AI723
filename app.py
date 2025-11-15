import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime
from pdf_handler import PDFHandler
from openrouter_client import OpenRouterClient

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

        # Chat history
        self.chat_history = []

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
        self.tab_pdf = self.tabview.add("Base de Connaissances PDF")
        self.tab_config = self.tabview.add("Configuration")

        # Setup each tab
        self.setup_chat_tab()
        self.setup_pdf_tab()
        self.setup_config_tab()

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "api_key": "",
            "model": "anthropic/claude-3.5-sonnet"
        }

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

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

    def setup_pdf_tab(self):
        """Setup the PDF management tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_pdf,
            text="Gestion de la Base de Connaissances PDF",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

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

        # Model section
        model_frame = ctk.CTkFrame(self.tab_config)
        model_frame.pack(fill="x", padx=20, pady=10)

        model_label = ctk.CTkLabel(
            model_frame,
            text="Modèle:",
            font=("Arial", 14)
        )
        model_label.pack(pady=5)

        self.model_entry = ctk.CTkEntry(
            model_frame,
            placeholder_text="Ex: anthropic/claude-3.5-sonnet",
            width=400
        )
        self.model_entry.pack(pady=5)
        if self.config.get("model"):
            self.model_entry.insert(0, self.config["model"])

        # Info label
        info_label = ctk.CTkLabel(
            model_frame,
            text="Exemples de modèles:\n"
                 "• anthropic/claude-3.5-sonnet\n"
                 "• openai/gpt-4-turbo\n"
                 "• google/gemini-pro",
            font=("Arial", 10),
            text_color="gray"
        )
        info_label.pack(pady=5)

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

    def send_message(self):
        """Send message to the LLM"""
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

        # Clear input
        self.message_input.delete("1.0", "end")

        # Add user message to chat
        self.add_message_to_chat("Vous", message)

        # Disable send button while processing
        self.send_button.configure(state="disabled", text="En cours...")
        self.root.update()

        # Build context from PDF knowledge
        context = self.build_context()

        # Send to OpenRouter
        try:
            response = self.openrouter_client.send_message(
                message,
                self.config["api_key"],
                self.config["model"],
                context,
                self.chat_history
            )

            # Add response to chat
            self.add_message_to_chat("Assistant", response)

            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la communication avec l'API:\n{str(e)}")
            self.add_message_to_chat("Système", f"Erreur: {str(e)}")

        # Re-enable send button
        self.send_button.configure(state="normal", text="Envoyer")

    def build_context(self):
        """Build context from PDF knowledge base"""
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
                    break

            # Add to knowledge base
            self.pdf_knowledge.append({
                "filename": filename,
                "content": text,
                "added_date": datetime.now().isoformat()
            })

            # Save to file
            self.save_pdf_knowledge()

            # Update display
            self.update_pdf_list()

            messagebox.showinfo("Succès", f"{filename} a été ajouté à la base de connaissances!")

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
            self.pdf_knowledge.pop(index)
            self.save_pdf_knowledge()
            self.update_pdf_list()

    def save_configuration(self):
        """Save the configuration"""
        api_key = self.api_key_entry.get().strip()
        model = self.model_entry.get().strip()

        if not api_key:
            messagebox.showerror("Erreur", "Veuillez entrer une clé API")
            return

        if not model:
            messagebox.showerror("Erreur", "Veuillez entrer un modèle")
            return

        self.config["api_key"] = api_key
        self.config["model"] = model

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
