import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileDownloader:
    @staticmethod
    def process(file_url: str, destination_path: str) -> bool:
        """
        Descarga un archivo desde una URL y lo guarda en la ruta especificada.

        Args:
            file_url (str): URL del archivo a descargar.
            destination_path (str): Ruta de destino donde se guardará el archivo descargado.

        Returns:
            bool: True si la descarga fue exitosa o en el caso que el archivo exista, False en caso contrario.
        """
        try:
            if os.path.exists(destination_path):
                logger.info("El archivo ya existe en la ruta de destino.")
                return True
            with requests.get(file_url) as response:
                response.raise_for_status()
                with open(destination_path, 'wb') as file:
                    file.write(response.content)
            logger.info("Descarga exitosa.")
            return True
        except FileNotFoundError as e:
            logger.error("No se pudo descargar el archivo. Archivo no encontrado en la URL proporcionada: "
                         f"{str(e)}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error HTTP al descargar el archivo: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Error de conexión al descargar el archivo: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Tiempo de espera agotado al descargar el archivo: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al realizar la solicitud HTTP: {str(e)}")
        except OSError as e:
            logger.error(f"Error al acceder o escribir en el archivo: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado al descargar el archivo: {str(e)}")
        return False
