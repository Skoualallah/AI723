import requests
import json


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.models_api_url = "https://openrouter.ai/api/v1/models"

        # Cache for model information
        self.model_info_cache = {}

        # Fallback model context limits (in tokens) if API call fails
        self.fallback_context_limits = {
            "anthropic/claude-3.5-sonnet": 200000,
            "anthropic/claude-3-opus": 200000,
            "anthropic/claude-3-sonnet": 200000,
            "anthropic/claude-3-haiku": 200000,
            "openai/gpt-4-turbo": 128000,
            "openai/gpt-4": 8192,
            "openai/gpt-3.5-turbo": 16385,
            "google/gemini-pro": 32768,
            "google/gemini-pro-1.5": 1000000,
            "meta-llama/llama-3.1-70b-instruct": 131072,
            "meta-llama/llama-3.1-8b-instruct": 131072,
        }

    def get_model_info(self, model, api_key):
        """
        Get model information from OpenRouter API

        Args:
            model: Model name
            api_key: OpenRouter API key

        Returns:
            Model information dictionary or None if error
        """
        # Check cache first
        if model in self.model_info_cache:
            return self.model_info_cache[model]

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

            response = requests.get(
                self.models_api_url,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                models_list = data.get('data', [])

                # Find the specific model
                for model_info in models_list:
                    if model_info.get('id') == model:
                        # Cache the model info
                        self.model_info_cache[model] = model_info
                        return model_info

            return None

        except Exception:
            # If API call fails, return None and use fallback
            return None

    def get_context_limit(self, model, api_key=None):
        """
        Get the context limit for a specific model from OpenRouter API

        Args:
            model: Model name
            api_key: OpenRouter API key (optional, needed for API call)

        Returns:
            Context limit in tokens
        """
        # Try to get from API if api_key is provided
        if api_key:
            model_info = self.get_model_info(model, api_key)
            if model_info:
                context_length = model_info.get('context_length')
                if context_length:
                    return context_length

        # Fallback to hardcoded values
        return self.fallback_context_limits.get(model, 8192)

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
            Dictionary with:
            - content: Response from the model
            - usage: Token usage information (prompt_tokens, completion_tokens, total_tokens)
            - model: Model used for the response
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
                result = {
                    'content': response_data['choices'][0]['message']['content'],
                    'usage': response_data.get('usage', {}),
                    'model': response_data.get('model', model)
                }
                return result
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
