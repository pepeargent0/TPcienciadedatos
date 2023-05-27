from typing import Union, List, Tuple
import logging
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error en el metodo {func.__name__}: {str(e)}")

    return wrapper


class DataExplorer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger('DataExplorer')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _validate_column(self, column):
        """
        Valida que la columna exista en el conjunto de datos.

        Args:
            column (str): El nombre de la columna a validar.

        Raises:
            ValueError: Si la columna no existe en el conjunto de datos.
        """
        if column not in self.data.columns:
            raise ValueError(f"La columna '{column}' no existe en el conjunto de datos.")

    def _validate_data(self, column):
        """
        Valida que los datos de la columna sean numéricos.

        Args:
            column (str): El nombre de la columna a validar.

        Raises:
            TypeError: Si los datos de la columna no son numéricos.
        """
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise TypeError(f"Los datos en la columna '{column}' no son numéricos.")

    @staticmethod
    def _validate_column_name(column_name):
        if not isinstance(column_name, str) or not column_name:
            raise ValueError("El nombre de la columna debe ser una cadena no vacía.")

    @staticmethod
    def _validate_x_y_columns(x_column, y_column):
        if not isinstance(x_column, str) or not isinstance(y_column, str):
            raise ValueError("Los nombres de las columnas deben ser cadenas.")
        if x_column == y_column:
            raise ValueError("Los nombres de las columnas para x y y deben ser diferentes.")

    @log_errors
    def plot_bar(self, x_columns: Union[str, List[str]], y_columns: Union[str, List[str]], title: str = '',
                 x_label: str = '', y_label: str = '', figsize: Tuple[float, float] = (8, 6)):
        """
        Genera un gráfico de barras.

        Args:
            x_columns (str): El nombre de la columna para el eje x.
            y_columns (str): El nombre de la columna para el eje y.
            title (str): El título del gráfico (opcional).
            x_label (str): La etiqueta del eje x (opcional).
            y_label (str): La etiqueta del eje y (opcional).
            figsize (Tuple[float, float]): Las dimensiones del contenedor del gráfico en pulgadas (opcional).
        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de alguna de las columnas no son numéricos.
        """
        if isinstance(x_columns, str):
            x_columns = [x_columns]
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        for y_column in y_columns:
            self._validate_column(y_column)
            self._validate_data(y_column)
        plt.figure(figsize=figsize)
        for y_column in y_columns:
            for x_column in x_columns:
                x_data = self.data[x_column]
                if np.issubdtype(x_data.dtype, np.datetime64):
                    x_data = x_data.dt.month_name() + ' ' + x_data.dt.year.astype(str)
                y_data = self.data[y_column]
                plt.bar(x_data, y_data, label=y_column)
        plt.xlabel(x_label if x_label else ', '.join(x_columns))
        plt.ylabel(y_label if y_label else ', '.join(y_columns))
        plt.title(title)
        plt.xticks(rotation=90)
        plt.legend()
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_line(self, x_columns: Union[str, List[str]], y_columns: Union[str, List[str]], title: str = '',
                  x_label: str = '', y_label: str = '', figsize: Tuple[float, float] = (8, 6)):
        """
        Genera un gráfico de líneas.

        Args:
            x_columns (Union[str, List[str]]): El nombre o lista de nombres de las columnas para el eje x.
            y_columns (Union[str, List[str]]): El nombre o lista de nombres de las columnas para el eje y.
            title (str): El título del gráfico (opcional).
            x_label (str): La etiqueta del eje x (opcional).
            y_label (str): La etiqueta del eje y (opcional).
            figsize (Tuple[float, float]): Las dimensiones del contenedor del gráfico en pulgadas (opcional).
        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de alguna de las columnas no son numéricos.
        """
        if isinstance(x_columns, str):
            x_columns = [x_columns]
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        for y_column in y_columns:
            self._validate_column(y_column)
            self._validate_data(y_column)
        plt.figure(figsize=figsize)
        for y_column in y_columns:
            for x_column in x_columns:
                x_data = self.data[x_column]
                if np.issubdtype(x_data.dtype, np.datetime64):
                    x_data = x_data.dt.month_name() + ' ' + x_data.dt.year.astype(str)
                y_data = self.data[y_column]
                plt.plot(x_data, y_data, label=y_column)
        plt.xlabel(x_label if x_label else ', '.join(x_columns))
        plt.ylabel(y_label if y_label else ', '.join(y_columns))
        plt.title(title)
        plt.xticks(rotation=90)
        plt.legend()
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_scatter(self, x_column: str, y_column: str, title: str = '',
                     x_label: str = '', y_label: str = '', figsize: Tuple[float, float] = (8, 6)):
        """
        Genera un gráfico de dispersión.

        Args:
            x_column (str): El nombre de la columna para el eje x.
            y_column (str): El nombres de la columna para el eje y.
            title (str): El título del gráfico (opcional).
            x_label (str): La etiqueta del eje x (opcional).
            y_label (str): La etiqueta del eje y (opcional).
            figsize (Tuple[float, float]): Las dimensiones del contenedor del gráfico en pulgadas (opcional).
        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de alguna de las columnas no son numéricos.
        """
        y_data = self.data[y_column]
        x_data = self.data[x_column]
        plt.figure(figsize=figsize)
        plt.scatter(x_data, y_data)
        plt.xlabel(x_label or x_column)
        plt.ylabel(y_label or y_column)
        if title == '':
            title = 'Genera un gráfico de dispersión: ' + x_column + ' vs ' + y_column
        plt.title(title)
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_histogram(self, column: str, bins=10):
        """
        Genera un histograma.

        Args:
            column (str): El nombre de la columna para generar el histograma.
            bins (int): El número de divisiones del histograma (por defecto 10).

        Raises:
            ValueError: Si la columna no existe en el conjunto de datos.
            TypeError: Si los datos de la columna no son numéricos.
        """
        self._validate_column(column)
        self._validate_data(column)
        data_column = self.data[column]
        plt.hist(data_column, bins=bins)
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.title('Histograma: ' + column)
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_boxplot(self, x: str, y: str):
        """
        Genera un diagrama de caja.

        Args:
            x (str): El nombre de la columna para el eje x.
            y (str): El nombre de la columna para el eje y.

        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de la columna y no son numéricos.
        """
        self._validate_column(x)
        self._validate_column(y)
        self._validate_data(y)
        sns.boxplot(x=x, y=y, data=self.data)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('Diagrama de Caja')
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_coor(self, columns: Union[str, List[str]], title: str = '', figsize: Tuple[float, float] = (8, 6)):
        """
        Genera un heatmap de correlación.

        Raises:
            ValueError: Si los datos no son numéricos.
        """
        corr_matrix = self.data[columns].corr()
        plt.figure(figsize=figsize)
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
        plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
        plt.title(title)
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_violinplot(self, x: str, y: str):
        """
        Genera un diagrama de violín.

        Args:
            x (str): El nombre de la columna para el eje x.
            y (str): El nombre de la columna para el eje y.

        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de la columna y no son numéricos.
        """
        self._validate_column(x)
        self._validate_column(y)
        self._validate_data(y)
        sns.violinplot(x=x, y=y, data=self.data)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('Diagrama de Violín')
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_area(self, x: str, y: str):
        """
        Genera un gráfico de área.

        Args:
            x (str): El nombre de la columna para el eje x.
            y (str): El nombre de la columna para el eje y.

        Raises:
            ValueError: Si alguna de las columnas no existe en el conjunto de datos.
            TypeError: Si los datos de la columna y no son numéricos.
        """
        self._validate_column(x)
        self._validate_column(y)
        self._validate_data(y)
        plt.fill_between(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('Gráfico de Área')
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_pie(self, column: str):
        """
        Genera un gráfico de torta.

        Args:
            column (str): El nombre de la columna para generar el gráfico de torta.

        Raises:
            ValueError: Si la columna no existe en el conjunto de datos.
            TypeError: Si los datos de la columna no son numéricos.
        """
        self._validate_column(column)
        data_counts = self.data[column].value_counts()
        plt.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%')
        plt.title('Gráfico de Torta')
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_outliers(self, column):
        """
        Genera un gráfico para identificar valores atípicos en una columna.

        Args:
            column (str): El nombre de la columna para la exploración de valores atípicos.

        Raises:
            ValueError: Si la columna no existe en el conjunto de datos.
            TypeError: Si los datos de la columna no son numéricos.
        """
        self._validate_column(column)
        self._validate_data(column)
        sns.boxplot(x=self.data[column])
        plt.xlabel(column)
        plt.title('Valores Atípicos: ' + column)
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_correlation(self):
        """
        Genera una matriz de correlación y un heatmap.

        Raises:
            ValueError: Si los datos no son numéricos.
        """
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap de Correlación')
        plt.show(block=False)
        plt.close()

    @log_errors
    def plot_distribution(self, column):
        """
        Genera un gráfico de distribución para una columna.

        Args:
            column (str): El nombre de la columna para la exploración de la distribución.

        Raises:
            ValueError: Si la columna no existe en el conjunto de datos.
            TypeError: Si los datos de la columna no son numéricos.
        """
        self._validate_column(column)
        self._validate_data(column)
        sns.histplot(self.data[column])
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.title('Distribución: ' + column)
        plt.show(block=False)
        plt.close()

    @log_errors
    def descriptive_statistics(self, column: str, title: str = '') -> None:
        """
        Genera un gráfico con las medidas de estadísticas descriptivas de esa columna.

        Args:
            column (str): Nombre de la columna.
            title (str): Título del gráfico (opcional).
        """
        self._validate_column(column)
        datos_columna = self.data[column]
        cantidad = datos_columna.count()
        media = round(datos_columna.mean(), 3)
        mediana = datos_columna.median()
        moda = datos_columna.mode().tolist()[0]
        minimo = datos_columna.min()
        maximo = datos_columna.max()
        desviacion_estandar = round(datos_columna.std(), 3)
        varianza = round(datos_columna.var(), 3)
        percentil_25 = datos_columna.quantile(0.25)
        percentil_50 = datos_columna.quantile(0.50)
        percentil_75 = datos_columna.quantile(0.75)

        # Crear los datos de la tabla
        datos = [
            ['Cantidad', cantidad],
            ['Media', media],
            ['Mediana', mediana],
            ['Moda', moda],
            ['Mínimo', minimo],
            ['Máximo', maximo],
            ['Desviación Estándar', desviacion_estandar],
            ['Varianza', varianza],
            ['Percentil 25', percentil_25],
            ['Percentil 50', percentil_50],
            ['Percentil 75', percentil_75]
        ]

        # Crear la tabla
        fig, ax = plt.subplots()
        ax.axis('off')  # Desactivar los ejes
        tabla = ax.table(cellText=datos, loc='center', colWidths=[0.3, 0.3], cellLoc='left')
        # Establecer el estilo de la tabla
        tabla.set_fontsize(14)
        tabla.scale(1.5, 1.5)  # Ajustar el tamaño de la tabla

        # Mostrar la figura
        plt.title(title or f"Estadísticas descriptivas para la columna '{column}'")
        plt.show(block=False)
