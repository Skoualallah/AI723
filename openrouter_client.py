import requests
import json


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def send_message(self, message, api_key, model, context="", chat_history=None):
        """
        Send a message to OpenRouter API

        Args:
            message: User message
            api_key: OpenRouter API key
            model: Model to use (e.g., "anthropic/claude-3.5-sonnet")
            context: Additional context from PDF knowledge base
            chat_history: Previous chat history

        Returns:
            Response from the model
        """
        if chat_history is None:
            chat_history = []

        # Build messages array
        messages = []

        # Add system message with context if available
        if context:
            messages.append({
                "role": "system",
                "content": f"{context}\n\nUtilise les informations ci-dessus pour répondre aux questions de l'utilisateur si elles sont pertinentes."
            })

        # Add chat history (keep last 10 messages to avoid token limit)
        if chat_history:
            messages.extend(chat_history[-10:])

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        # Prepare request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages
        }

        try:
            # Send request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=60
            )

            # Check for errors
            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                error_message = error_data.get('error', {}).get('message', response.text)
                raise Exception(f"Erreur API (code {response.status_code}): {error_message}")

            # Parse response
            response_data = response.json()

            # Extract message content
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                raise Exception("Réponse API invalide: pas de choix disponible")

        except requests.exceptions.Timeout:
            raise Exception("La requête a expiré. Veuillez réessayer.")
        except requests.exceptions.ConnectionError:
            raise Exception("Erreur de connexion. Vérifiez votre connexion internet.")
        except json.JSONDecodeError:
            raise Exception("Erreur lors du décodage de la réponse API")
        except Exception as e:
            if "Erreur API" in str(e) or "requête a expiré" in str(e) or "connexion" in str(e):
                raise
            raise Exception(f"Erreur inattendue: {str(e)}")
