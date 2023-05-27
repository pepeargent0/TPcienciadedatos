import pandas as pd
import numpy as np
import logging


class DataCleaner:
    def __init__(self, data_frame: pd.DataFrame, value_nan: str = "mode", outlier: str = "all"):
        """
        Inicializa el objeto DataCleaner.

        Args:
            data_frame (pd.DataFrame): El marco de datos a limpiar.
            value_nan (str, opcional): Valor a utilizar para reemplazar los valores NaN. Puede ser 'mean', 'median',
                                       'mode', 'previous', 'next', 'empty' o 'interpolate'. Por defecto es 'mode'.
            outlier (str, opcional): Tipo de valores atípicos a considerar. Puede ser 'all', 'upper', 'lower',
                                     'first' o 'last'. Por defecto es 'all'.
        """
        if data_frame is None or data_frame.empty:
            raise ValueError("El marco de datos de entrada no puede ser None o estar vacío.")
        self.data_frame = data_frame
        self.value_nan = value_nan
        self.outlier = outlier
        self.logger = logging.getLogger("DataCleaner")
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def get_cleaned_data(self):
        """
        Obtiene el marco de datos limpio.

        Returns:
            pd.DataFrame: El marco de datos limpio.
        """
        columns = [column.capitalize() for column in self.data_frame.columns]
        self.data_frame.columns = columns
        return self.data_frame

    def detect_outliers(self, column, threshold=1.5):
        """
        Detecta y reemplaza los valores atípicos en una columna numérica.

        Args:
            column (str): Nombre de la columna a procesar.
            threshold (float, opcional): Umbral para definir los valores atípicos. Por defecto es 1.5.
        """
        try:
            q1 = np.percentile(self.data_frame[column], 25)
            q3 = np.percentile(self.data_frame[column], 75)
            iqr = q3 - q1
            lower_limit = q1 - threshold * iqr
            upper_limit = q3 + threshold * iqr

            outlier_operations = {
                'all': (self.data_frame[column] < lower_limit) | (self.data_frame[column] > upper_limit),
                'upper': self.data_frame[column] > upper_limit,
                'lower': self.data_frame[column] < lower_limit,
                'first': self.data_frame[column] == self.data_frame[column].min(),
                'last': self.data_frame[column] == self.data_frame[column].max()
            }

            outlier_mask = outlier_operations.get(self.outlier, False)
            self.data_frame[column] = np.where(outlier_mask, self.data_frame[column].median(), self.data_frame[column])

        except Exception as e:
            self.logger.error(f"Error al detectar valores atípicos en la columna {column}: {str(e)}")

    def remove_duplicate_rows(self):
        """
        Elimina las filas duplicadas del marco de datos.
        """
        try:
            self.data_frame.drop_duplicates(inplace=True)
        except Exception as e:
            self.logger.error(f"Error al eliminar filas duplicadas: {str(e)}")

    def impute_nan_values(self, column: str):
        """
        Imputa los valores NaN en una columna numérica.

        Args:
            column (str): Nombre de la columna a procesar.
        """
        try:
            if self.value_nan == 'interpolate':
                self.data_frame[column].interpolate(method='linear', inplace=True)
            else:
                impute_value = {
                    'mean': self.data_frame[column].mean(),
                    'median': self.data_frame[column].median(),
                    'mode': self.data_frame[column].mode()[0],
                    'previous': self.data_frame[column].ffill(),
                    'next': self.data_frame[column].bfill(),
                    'empty': ''
                }.get(self.value_nan, 0)
                self.data_frame[column].fillna(impute_value, inplace=True)

        except Exception as e:
            self.logger.error(f"Error al imputar valores NaN en la columna {column}: {str(e)}")

    def clean_numerical_column(self, column: str):
        """
        Realiza la limpieza de una columna numérica.

        Args:
            column (str): Nombre de la columna a procesar.
        """
        try:
            self.detect_outliers(column)
            if self.data_frame[column].isnull().any():
                self.impute_nan_values(column)

            if (self.data_frame[column] % 1 == 0).all():
                self.data_frame[column] = self.data_frame[column].astype(int)

        except Exception as e:
            self.logger.error(f"Error al limpiar la columna numérica {column}: {str(e)}")

    def clean_text_column(self, column: str):
        """
        Realiza la limpieza de una columna de texto.

        Args:
            column (str): Nombre de la columna a procesar.
        """
        try:
            self.data_frame[column].str.replace(' ', '', regex=True)
            if column == 'tasa_de_empleo(%)':
                self.data_frame[column] = self.data_frame[column].str.replace('.', ',', regex=False)
            else:
                self.data_frame[column] = self.data_frame[column].str.replace('.', '', regex=False)
            self.data_frame[column] = self.data_frame[column].str.replace(',', '.', regex=False).astype(float)
            self.clean_numerical_column(column)

        except Exception as e:
            self.logger.error(f"Error al limpiar la columna de texto {column}: {str(e)}")

    def convert_negative_values_to_nan(self, column: str):
        """
        Convierte los valores negativos en una columna a NaN.

        Args:
            column (str): Nombre de la columna a procesar.
        """
        try:
            self.data_frame[column] = self.data_frame[column].apply(lambda value: value if value >= 0 else None)
        except Exception as e:
            self.logger.error(f"Error al convertir los valores negativos a NaN en la columna {column}: {str(e)}")

    def clean_date_column(self, column: str):
        """
        Realiza la limpieza de una columna de fechas.

        Args:
            column (str): Nombre de la columna a procesar.
        """
        try:
            self.data_frame[column] = pd.to_datetime(self.data_frame[column], format='%Y-%m-%d')
            self.data_frame.dropna(subset=[column], inplace=True)
            self.data_frame = self.data_frame.loc[self.data_frame[column] >= pd.Timestamp.min]

        except Exception as e:
            self.logger.error(f"Error al limpiar la columna de fecha {column}: {str(e)}")

    def detect_and_clean_outliers(self, column: str, threshold: float = 1.5):
        """
        Detecta y limpia los valores atípicos en una columna numérica.

        Args:
            column (str): Nombre de la columna a procesar.
            threshold (float, opcional): Umbral para definir los valores atípicos. Por defecto es 1.5.
        """
        try:
            self.detect_outliers(column, threshold)
            self.clean_numerical_column(column)

        except Exception as e:
            self.logger.error(f"Error al detectar y limpiar valores atípicos en la columna {column}: {str(e)}")

    def cleanse_data_columns(self):
        """
        Realiza la limpieza de todas las columnas de datos en el marco de datos.
        """
        try:
            float_columns = self.data_frame.select_dtypes(include=['float']).columns
            object_columns = self.data_frame.select_dtypes(include=['object']).columns
            date_columns = self.data_frame.select_dtypes(include=['datetime']).columns

            for column in float_columns:
                self.detect_and_clean_outliers(column)

            for column in object_columns:
                pattern = r'[a-zA-Z]{2}'
                matches = self.data_frame[column].str.contains(pattern, regex=True)
                if matches.any():
                    self.data_frame[column] = self.data_frame[column].str.capitalize()
                else:
                    self.clean_text_column(column)

            for column in date_columns:
                self.clean_date_column(column)

        except Exception as e:
            self.logger.error(f"Error al limpiar las columnas de datos: {str(e)}")

    def save_cleaned_data(self, output_filename: str):
        """
        Guarda los datos limpios en un archivo en formato Parquet.

        Args:
            output_filename (str): Nombre del archivo de salida.
        """
        try:
            columns = [column.capitalize() for column in self.data_frame.columns]
            self.data_frame.columns = columns
            self.data_frame.to_parquet(output_filename, index=False)
            self.logger.info(f"Los datos limpios se han guardado en el archivo {output_filename}")
            self.logger.info(f"UUID único generado: {self.uuid}")

        except Exception as e:
            self.logger.error(f"Error al guardar los datos limpios: {str(e)}")