import pandas as pd
import json

def limpiar_dataset(archivo_entrada, archivo_salida):
    print("📂 Cargando dataset...")
    df = pd.read_csv(archivo_entrada, encoding='utf-8')
    
    print(f"📊 Registros originales: {len(df)}")
    print(f"📋 Columnas: {list(df.columns)}")
    
    print("\n🧹 Limpiando datos...")
    df_limpio = df.dropna(subset=['texto', 'tema'])
    
    registros_antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates(subset=['texto'], keep='first')
    duplicados_eliminados = registros_antes - len(df_limpio)
    
    print(f"🗑️  Duplicados eliminados: {duplicados_eliminados}")
    print(f"✅ Registros finales: {len(df_limpio)}")
    
    print("\n📊 Distribución por tema:")
    temas = df_limpio['tema'].value_counts()
    for tema, cantidad in temas.items():
        print(f"  • {tema}: {cantidad}")
    
    print("\n😊 Distribución por sentimiento:")
    sentimientos = df_limpio['sentimiento'].value_counts()
    for sentimiento, cantidad in sentimientos.items():
        print(f"  • {sentimiento}: {cantidad}")
    
    df_limpio.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f"\n💾 Dataset limpio guardado en: {archivo_salida}")
    
    reporte = {
        "registros_originales": len(df),
        "registros_finales": len(df_limpio),
        "duplicados_eliminados": duplicados_eliminados,
        "temas": temas.to_dict(),
        "sentimientos": sentimientos.to_dict()
    }
    
    with open('reporte_limpieza.json', 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print("📝 Reporte guardado en: reporte_limpieza.json")
    
    return df_limpio


if __name__ == "__main__":
    df_limpio = limpiar_dataset(
        archivo_entrada='dataset_sintetico_5000_ampliado.csv',
        archivo_salida='dataset_limpio.csv'
    )
