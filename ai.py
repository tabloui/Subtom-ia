import requests
import json

def generate_image(prompt: str) -> str:
    """
    Genera una imagen utilizando la API pública/endpoint de DeepAI.
    Devuelve la URL de la imagen generada o lanza una excepción si falla.
    """
    url = "https://api.deepai.org/api/text2img"
    
    # DeepAI suele requerir form-data básico o API key para cuentas pro,
    # pero podemos usar el endpoint estándar o una alternativa robusta.
    # Nota: Usamos una clave de prueba pública o standard si está disponible,
    # o simulamos la petición al servicio de DeepAI text2img.
    payload = {
        'text': prompt,
    }
    
    # En DeepAI a veces se usa una api-key 'quickstart-xxxxxxxx' o similar,
    # o se usa el endpoint público gratuito con 'api-key': 'api-key'.
    headers = {
        'api-key': 'quickstart-hacked' # O la clave que corresponda de DeepAI
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("output_url", "")
        else:
            raise Exception(f"Error en DeepAI: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error generando imagen con DeepAI: {e}")
        raise e
