import google.generativeai as genai
import json


class GoogleClient:
    """Client for Google AI Studio (Gemini) API"""

    def __init__(self):
        self.api_key = None
        self.configured = False

    def configure(self, api_key):
        """
        Configure the Google AI client with API key

        Args:
            api_key: Google AI Studio API key
        """
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.configured = True

    def send_message(self, model_name, message, system_prompt=None, use_structured_output=False):
        """
        Send a message to a Gemini model

        Args:
            model_name: Name of the Gemini model (e.g., 'gemini-pro', 'gemini-1.5-pro')
            message: The user message to send
            system_prompt: Optional system prompt/context
            use_structured_output: Whether to request structured JSON output

        Returns:
            Dictionary with response data including content, usage stats, etc.
        """
        if not self.configured:
            raise Exception("Google AI client not configured. Please set your API key first.")

        try:
            # Clean model name (remove provider prefix if present)
            if '/' in model_name:
                model_name = model_name.split('/')[-1]

            # Initialize the model
            model = genai.GenerativeModel(model_name)

            # Prepare the full prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {message}"
            else:
                full_prompt = message

            # Add structured output instruction if needed
            if use_structured_output:
                full_prompt += "\n\nPlease respond in JSON format with the following structure: {\"explanation\": \"your detailed explanation here\", \"final_answer\": \"your answer here\", \"final_answer_letter\": \"single letter (A, B, C, etc.)\"}"

            # Generate response
            response = model.generate_content(full_prompt)

            # Extract the response text
            response_text = response.text

            # Parse structured output if requested
            answer_letter = None
            if use_structured_output:
                try:
                    # Try to extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_json = json.loads(json_str)
                        answer_letter = parsed_json.get('final_answer_letter', '?')
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, try to extract letter from text
                    answer_letter = '?'

            # Calculate usage statistics
            # Note: Google AI SDK provides usage metadata
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count

            # Get model context length (approximate values)
            context_lengths = {
                'gemini-pro': 32768,
                'gemini-1.5-pro': 1000000,
                'gemini-1.5-pro-latest': 1000000,
                'gemini-1.5-flash': 1000000,
                'gemini-1.5-flash-latest': 1000000,
                'gemini-2.0-flash-exp': 1000000,
            }
            context_length = context_lengths.get(model_name, 32768)

            # Calculate context usage percentage
            context_usage = (total_tokens / context_length * 100) if context_length > 0 else 0

            return {
                'content': response_text,
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                },
                'context_length': context_length,
                'context_usage': min(context_usage, 100),  # Cap at 100%
                'answer_letter': answer_letter,
                'model': model_name,
                'provider': 'google'
            }

        except Exception as e:
            error_message = str(e)

            # Check for specific error types
            if "API_KEY_INVALID" in error_message or "invalid API key" in error_message.lower():
                raise Exception("Clé API Google invalide. Veuillez vérifier votre clé API dans la configuration.")
            elif "quota" in error_message.lower():
                raise Exception(f"Quota dépassé: {error_message}")
            elif "not found" in error_message.lower():
                raise Exception(f"Modèle non trouvé: {model_name}. Vérifiez le nom du modèle.")
            else:
                raise Exception(f"Erreur Google AI: {error_message}")

    def list_models(self):
        """
        List available Gemini models

        Returns:
            List of available model names
        """
        if not self.configured:
            return []

        try:
            models = genai.list_models()
            model_names = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    # Extract model name (remove 'models/' prefix)
                    name = model.name.replace('models/', '')
                    model_names.append(name)
            return model_names
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
