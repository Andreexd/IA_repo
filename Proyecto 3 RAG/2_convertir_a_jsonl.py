import pandas as pd
import json

def csv_a_jsonl(archivo_csv, archivo_jsonl):
    print("📂 Cargando dataset limpio...")
    df = pd.read_csv(archivo_csv, encoding='utf-8')
    
    print(f"📊 Procesando {len(df)} registros...")
    
    documentos = []
    
    for idx, row in df.iterrows():
        documento = {
            "id": f"doc_{row['id']}",
            "content": row['texto'],
            "metadata": {
                "tema": row['tema'],
                "sentimiento": row['sentimiento'],
                "fecha": row['fecha'],
                "likes": int(row['likes']),
                "reposts": int(row['reposts']),
                "usuario": row['usuario']
            },
            "title": f"{row['tema']} - {row['sentimiento']}",
            "date": row['fecha']
        }
        
        documentos.append(documento)
    
    with open(archivo_jsonl, 'w', encoding='utf-8') as f:
        for doc in documentos:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Archivo JSONL creado: {archivo_jsonl}")
    print(f"📝 Total de documentos: {len(documentos)}")
    
    archivo_json = archivo_jsonl.replace('.jsonl', '.json')
    with open(archivo_json, 'w', encoding='utf-8') as f:
        json.dump(documentos, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Archivo JSON creado: {archivo_json}")
    
    print("\n📄 Ejemplo de documento:")
    print(json.dumps(documentos[0], indent=2, ensure_ascii=False))
    
    return documentos


if __name__ == "__main__":
    documentos = csv_a_jsonl(
        archivo_csv='dataset_limpio.csv',
        archivo_jsonl='dataset_para_llm.jsonl'
    )
