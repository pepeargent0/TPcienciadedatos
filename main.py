import pandas as pd
from datacleaner.datacleaner import DataCleaner
from downloader.download import FileDownloader
from dataexplorer.dataexplorer import DataExplorer

cleaned_csv_filename = 'datos_limpios_ej1.csv'
csv_filename = 'provicias_datos_2022.csv'
url = 'https://raw.githubusercontent.com/pepeargent0/TPcienciadedatos/main/provicias_datos_2022.csv'
# enunciado 1
if FileDownloader().process(url, csv_filename):
    data_cleaner = DataCleaner(
        pd.read_csv(
            csv_filename,
            sep=";",
            thousands='.',
            skiprows=3
        ),
        value_nan='median',
        outlier='last'
    )
    data_cleaner.remove_duplicate_rows()
    data_cleaner.cleanse_data_columns()
    data_cleaner_1 = data_cleaner.get_cleaned_data()
    exploration = DataExplorer(data_cleaner_1)
    exploration.plot_bar(
        'Provincia',
        'Hogares (miles)',
        figsize=(8, 12)
    )
    # Gráfico de dispersión: Población vs. Hogares
    exploration.plot_scatter(
        'Población (miles)',
        'Hogares (miles)',
        x_label='Población (miles)',
        y_label='Hogares (miles)',
        title='Relación entre Población y Hogares'
    )
    # Gráfico de barras: Tasa de empleo por provincia
    exploration.plot_bar(
        'Provincia',
        'Tasa de empleo (%)',
        x_label='Provincia',
        y_label='Tasa de empleo (%)',
        title='Tasa de empleo por Provincia',
        figsize=(8, 12)
    )
    # Mapa de calor de correlaciones
    corr_matrix = [
        'Población (miles)',
        'Hogares (miles)',
        'Ingresos laborales (miles pesos)',
        'Tasa de empleo (%)', 'Superficie (km2)',
        'Exportaciones (mill usd)'
    ]
    exploration.plot_coor(
        corr_matrix,
        title='Matriz de Correlación',
        figsize=(8, 12)
    )
    # Relación entre "Población (miles)" y "Ingresos laborales (miles pesos)":
    exploration.plot_scatter(
        'Población (miles)',
        'Ingresos laborales (miles pesos)',
        x_label='Población (miles)',
        y_label='Ingresos laborales (miles pesos)',
        title='Relación entre Población e Ingresos laborales'
    )

    # Relación entre "Superficie (km2)" y "Exportaciones (mill usd)":
    exploration.plot_scatter(
        'Superficie (km2)',
        'Exportaciones (mill usd)',
        x_label='Superficie (km2)',
        y_label='Exportaciones (mill usd)',
        title='Relación entre Superficie y Exportaciones'
    )
    # Relación entre "Tasa de empleo (%)" y "Ingresos laborales (miles pesos)":
    exploration.plot_scatter(
        'Tasa de empleo (%)',
        'Ingresos laborales (miles pesos)',
        title='Relación entre Tasa de empleo e Ingresos laborales'
    )
    # Relación entre "Superficie (km2)" y "Exportaciones (mill usd)" utilizando un gráfico de barras:
    exploration.plot_bar(
        'Provincia',
        ['Superficie (km2)', 'Exportaciones (mill usd)'],
        x_label='Provincia',
        y_label='Valor',
        title='Superficie vs. Exportaciones por Provincia'
    )
    # GUARDADO DEL ARCHIVO PARQUET
    data_cleaner.save_cleaned_data('ejercicio1_limpio.parquet')
# EJERCICIO 2
xlsx_filename = 'tc_turistas.xlsx'
url = 'https://raw.githubusercontent.com/pepeargent0/TPcienciadedatos/main/tc_turistas.xlsx'
if FileDownloader().process(url, xlsx_filename):
    xlsx_filename = 'tc_turistas.xlsx'
    url = 'https://raw.githubusercontent.com/pepeargent0/TPcienciadedatos/main/tc_turistas.xlsx'
    cleaned_money = DataCleaner(
        pd.read_excel(
            xlsx_filename,
            sheet_name='tipos de cambio',
            thousands='.'
        ),
        'previous'
    )
    cleaned_money.remove_duplicate_rows()
    cleaned_money.cleanse_data_columns()
    # Comienza la limpieza de la hoja llegadas de turistas
    cleaned_tourist = DataCleaner(
        pd.read_excel(
            xlsx_filename,
            sheet_name='llegadas de turistas',
            thousands='.'
        ),
        'median',
        'upper'
    )
    cleaned_tourist.remove_duplicate_rows()
    cleaned_tourist.cleanse_data_columns()
    pd.set_option('display.float_format', '{:.2f}'.format)
    cleaned_tourist.data_frame['FECHA'] = pd.to_datetime(
        cleaned_tourist.data_frame['date'],
        origin='1899-12-30',
        unit='D'
    )
    merge_dataset = pd.merge(
        cleaned_money.data_frame[['FECHA', 'DOLAR Oficial', 'DOLAR Blue']],
        cleaned_tourist.data_frame[['Bolivia', 'Brasil', 'Chile', 'Paraguay', 'Uruguay', 'FECHA']],
        on='FECHA'
    ).round(2)
    cleaned_tourist.data_frame = merge_dataset
    exploration = DataExplorer(cleaned_tourist.data_frame)
    # Medidas De Estadisticas Descriptiva por Pais
    exploration.descriptive_statistics('Bolivia')
    exploration.descriptive_statistics('Brasil')
    exploration.descriptive_statistics('Chile')
    exploration.descriptive_statistics('Paraguay')
    exploration.descriptive_statistics('Uruguay')
    # Medidas De Estadisticas Descriptiva por Precios del Dolar
    exploration.descriptive_statistics('DOLAR Oficial')
    exploration.descriptive_statistics('DOLAR Blue', title='Variacion Del Dolar BLue')
    # Graficar el precio del dólar oficial y el dólar blue a lo largo del tiempo
    exploration.plot_line(
        'FECHA', ['DOLAR Oficial', 'DOLAR Blue'],
        title='Comparación del precio del dólar oficial y el dólar blue',
        x_label='Fecha',
        y_label='Precio del dólar',
        figsize=(10, 14)
    )
    # Graficar la cantidad de turistas de Bolivia en Argentina a lo largo del tiempo
    exploration.plot_line(
        'FECHA',
        'Bolivia',
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Bolivia en Argentina',
        figsize=(10, 14)
    )
    # Graficar la cantidad de turistas de Brasil en Argentina a lo largo del tiempo
    exploration.plot_line(
        'FECHA',
        'Brasil',
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Brasil en Argentina',
        figsize=(10, 14)
    )
    # Graficar la cantidad de turistas de Chile en Argentina a lo largo del tiempo
    exploration.plot_line(
        'FECHA',
        'Chile',
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Chile en Argentina',
        figsize=(10, 14)
    )
    # Graficar la cantidad de turistas de Uruguay en Argentina a lo largo del tiempo
    exploration.plot_line(
        'FECHA',
        'Uruguay',
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Uruguay en Argentina',
        figsize=(10, 14)
    )
    # Graficar la cantidad de turistas de Paraguay en Argentina a lo largo del tiempo
    exploration.plot_line(
        'FECHA',
        'Paraguay',
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Paraguay en Argentina',
        figsize=(10, 14)
    )
    # Graficar la relación entre el precio del dólar oficial y la cantidad de turistas de Uruguay en Argentina
    exploration.plot_scatter(
        'DOLAR Oficial',
        'Uruguay',
        x_label='Precio del dólar oficial',
        y_label='Cantidad de turistas de Uruguay',
        title='Relación entre el precio del dólar oficial y turistas de Uruguay en Argentina'
    )
    # Filtrar los datos para un mes específico
    mes_especifico = cleaned_tourist.data_frame[cleaned_tourist.data_frame['FECHA'].dt.month == 1]  # Ejemplo: enero
    # Graficar la cantidad de turistas de cada país en Argentina en un mes específico
    exploration_coustomer = DataExplorer(mes_especifico)
    exploration_coustomer.plot_line(
        'FECHA',
        ['Bolivia', 'Brasil', 'Chile', 'Paraguay', 'Uruguay'],
        x_label='Fecha',
        y_label='Cantidad de turistas',
        title='Turistas de Paraguay en Argentina',
        figsize=(10, 14)
    )
    # GUARDADO DEL ARCHIVO PARQUET
    cleaned_tourist.save_cleaned_data('ejercicio2_limpio.parquet')
