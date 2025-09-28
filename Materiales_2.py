# -*- coding: utf-8 -*-
"""
DESCARGADOR MATERIALS PROJECT - VERSIÓN OFICIAL BASADA EN EJEMPLOS DE GITHUB
Basado en: https://docs.materialsproject.org/downloading-data/using-the-api/examples
Compatible con mp-api >= 0.37, Python 3.8+
CONFIGURADO PARA DESCARGAR MUCHOS MÁS MATERIALES
"""

# INSTALAR ANTES DE EJECUTAR:
# pip install mp-api pymatgen pandas numpy

import pandas as pd
import numpy as np
from mp_api.client import MPRester
import warnings
warnings.filterwarnings('ignore')

# TU API KEY
API_KEY = "ALMHMDStK5zs2C7u850Srnhv0UKNCWbC"     #PONER LA API DE CADA UNO SEGUN LA WEB

# CONFIGURACIÓN DE DESCARGA
MAX_MATERIALES = 10000  # Cambia aquí el número que quieras (5000, 10000, 20000...)

def main():
    print("DESCARGADOR MATERIALS PROJECT - VERSIÓN OFICIAL")
    print("=" * 60)
    print(f"Configurado para descargar hasta {MAX_MATERIALES:,} materiales")
    
    with MPRester(API_KEY) as mpr:
        print("Conectado a Materials Project")
        
        # EJEMPLO OFICIAL: Band gaps para materiales con Si y O
        # Basado en: https://docs.materialsproject.org/downloading-data/using-the-api/examples
        print("Descargando materiales semiconductores...")
        print("Esto puede tomar varios minutos para grandes cantidades...")
        
        docs = mpr.materials.summary.search(
            band_gap=(0.001, None),  # Semiconductores
            fields=["material_id", "formula_pretty", "band_gap", 
                   "formation_energy_per_atom", "energy_above_hull",
                   "density", "nsites", "volume", "elements", 
                   "composition", "symmetry"],
            chunk_size=1000  # Aumentar chunk_size para descargas más rápidas
        )
        
        print(f"Encontrados {len(docs):,} materiales en total")
        
        # Determinar cuántos procesar
        materiales_a_procesar = min(len(docs), MAX_MATERIALES)
        print(f"Procesando {materiales_a_procesar:,} materiales...")
        
        # Procesar datos según la estructura oficial de la API
        registros = []
        errores = 0
        
        for i, doc in enumerate(docs[:materiales_a_procesar]):
            try:
                # ACCESO OFICIAL POR ATRIBUTO (no diccionario)
                # Como se muestra en los ejemplos oficiales de GitHub
                registro = {
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "band_gap": float(doc.band_gap) if doc.band_gap is not None else np.nan,
                    "formation_energy_per_atom": float(doc.formation_energy_per_atom) if doc.formation_energy_per_atom is not None else np.nan,
                    "energy_above_hull": float(doc.energy_above_hull) if doc.energy_above_hull is not None else np.nan,
                    "density": float(doc.density) if doc.density is not None else np.nan,
                    "nsites": int(doc.nsites) if doc.nsites is not None else 0,
                    "volume": float(doc.volume) if doc.volume is not None else np.nan,
                }
                
                # Información de simetría (de symmetry field)
                if hasattr(doc, 'symmetry') and doc.symmetry:
                    registro["crystal_system"] = getattr(doc.symmetry, 'crystal_system', '')
                    registro["spacegroup_number"] = getattr(doc.symmetry, 'number', np.nan)
                    registro["spacegroup_symbol"] = getattr(doc.symmetry, 'symbol', '')
                else:
                    registro["crystal_system"] = ''
                    registro["spacegroup_number"] = np.nan
                    registro["spacegroup_symbol"] = ''
                
                # Volumen por átomo
                if registro['volume'] and registro['nsites'] > 0:
                    registro['volume_per_atom'] = registro['volume'] / registro['nsites']
                else:
                    registro['volume_per_atom'] = np.nan
                
                # Elementos y composición
                elements = list(doc.elements) if doc.elements else []
                registro["n_elements"] = len(elements)
                registro["is_binary"] = len(elements) == 2
                registro["is_ternary"] = len(elements) == 3
                registro["is_quaternary"] = len(elements) == 4
                
                # Propiedades atómicas básicas usando pymatgen
                try:
                    from pymatgen.core import Element
                    if elements:
                        masses = [Element(el).atomic_mass for el in elements]
                        electronegativities = [Element(el).X for el in elements if Element(el).X]
                        radii = [Element(el).atomic_radius for el in elements if Element(el).atomic_radius]
                        
                        registro["avg_atomic_mass"] = np.mean(masses) if masses else np.nan
                        registro["min_atomic_mass"] = np.min(masses) if masses else np.nan
                        registro["max_atomic_mass"] = np.max(masses) if masses else np.nan
                        registro["avg_electronegativity"] = np.mean(electronegativities) if electronegativities else np.nan
                        registro["min_electronegativity"] = np.min(electronegativities) if electronegativities else np.nan
                        registro["max_electronegativity"] = np.max(electronegativities) if electronegativities else np.nan
                        registro["electronegativity_range"] = (np.max(electronegativities) - np.min(electronegativities)) if len(electronegativities) > 1 else 0
                        registro["avg_atomic_radius"] = np.mean(radii) if radii else np.nan
                        registro["avg_ionic_radius"] = np.nan
                    else:
                        registro["avg_atomic_mass"] = np.nan
                        registro["min_atomic_mass"] = np.nan
                        registro["max_atomic_mass"] = np.nan
                        registro["avg_electronegativity"] = np.nan
                        registro["min_electronegativity"] = np.nan
                        registro["max_electronegativity"] = np.nan
                        registro["electronegativity_range"] = np.nan
                        registro["avg_atomic_radius"] = np.nan
                        registro["avg_ionic_radius"] = np.nan
                except:
                    # Si falla pymatgen, usar valores nan
                    registro["avg_atomic_mass"] = np.nan
                    registro["min_atomic_mass"] = np.nan
                    registro["max_atomic_mass"] = np.nan
                    registro["avg_electronegativity"] = np.nan
                    registro["min_electronegativity"] = np.nan
                    registro["max_electronegativity"] = np.nan
                    registro["electronegativity_range"] = np.nan
                    registro["avg_atomic_radius"] = np.nan
                    registro["avg_ionic_radius"] = np.nan
                
                # Clasificación de elementos
                metals = {'Li','Na','K','Rb','Cs','Be','Mg','Ca','Sr','Ba','Al','Ga','In','Tl','Sn','Pb','Bi','Fe','Co','Ni','Cu','Zn','Ag','Cd','Au','Hg','Ti','V','Cr','Mn','Y','Zr','Nb','Mo','Ru','Rh','Pd','Hf','Ta','W','Re','Os','Ir','Pt'}
                metalloids = {'B','Si','Ge','As','Sb','Te','Po'}
                nonmetals = {'H','He','C','N','O','F','Ne','P','S','Cl','Ar','Se','Br','Kr','I','Xe','At','Rn'}
                
                registro["has_metal"] = any(el in metals for el in elements)
                registro["has_metalloid"] = any(el in metalloids for el in elements)
                registro["has_nonmetal"] = any(el in nonmetals for el in elements)
                registro["n_metals"] = sum(1 for el in elements if el in metals)
                registro["n_metalloids"] = sum(1 for el in elements if el in metalloids)
                registro["n_nonmetals"] = sum(1 for el in elements if el in nonmetals)
                
                # Fracciones atómicas (inicializar en 0)
                elementos_importantes = ['H','Li','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ca','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Ba','La','Ce','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi']
                for el in elementos_importantes:
                    # Inicializar todas las fracciones en 0
                    registro[f'frac_{el}'] = 0.0
                
                # Si hay información de composición, actualizar
                if hasattr(doc, 'composition') and doc.composition:
                    composition_dict = dict(doc.composition)
                    for el in elementos_importantes:
                        if el in composition_dict:
                            registro[f'frac_{el}'] = float(composition_dict[el])
                
                # Variables target según el PDF
                bg = registro["band_gap"]
                ehull = registro["energy_above_hull"]
                
                registro["is_semiconductor"] = (0.1 < bg < 4.0) if not np.isnan(bg) else False
                registro["is_photovoltaic"] = (0.8 <= bg <= 2.2) if not np.isnan(bg) else False
                registro["is_stable"] = (ehull < 0.1) if not np.isnan(ehull) else False
                
                registros.append(registro)
                
                # Mostrar progreso cada 500 materiales
                if (i + 1) % 500 == 0:
                    porcentaje = ((i + 1) / materiales_a_procesar) * 100
                    print(f"Procesados {i + 1:,}/{materiales_a_procesar:,} materiales ({porcentaje:.1f}%) - Errores: {errores}")
                    
            except Exception as e:
                errores += 1
                if errores <= 10:  # Solo mostrar los primeros 10 errores
                    print(f"Error procesando material {i}: {e}")
                continue
        
        # Crear DataFrame
        if registros:
            df = pd.DataFrame(registros)
            print(f"\nDataFrame creado: {len(df):,} filas, {len(df.columns)} columnas")
            print(f"Errores durante procesamiento: {errores}")
            
            # Verificar que tenemos las columnas necesarias
            print(f"Columnas principales: {[col for col in ['material_id', 'formula', 'band_gap', 'crystal_system'] if col in df.columns]}")
            
            # Guardar archivos CSV
            print("\nGuardando archivos CSV...")
            
            # Dataset principal
            df.to_csv("materials_project_dataset.csv", index=False)
            print(f"materials_project_dataset.csv ({len(df):,} materiales)")
            
            # Verificar que la columna band_gap existe antes de usarla
            if 'band_gap' in df.columns:
                # Caso 1: Semiconductores
                df_semi = df[df['band_gap'].notna()].copy()
                df_semi.to_csv("caso1_semiconductores.csv", index=False)
                print(f"caso1_semiconductores.csv ({len(df_semi):,} materiales)")
                
                # Caso 2: Fotovoltaicos
                df_foto = df[df['band_gap'].between(0.5, 3.0)].copy()
                df_foto.to_csv("caso2_fotovoltaicos.csv", index=False)
                print(f"caso2_fotovoltaicos.csv ({len(df_foto):,} materiales)")
                
                # Caso 4: Regresión Band Gap
                df_bg = df[(df['band_gap'].notna()) & (df['band_gap'] > 0)].copy()
                df_bg.to_csv("caso4_bandgap_regresion.csv", index=False)
                print(f"caso4_bandgap_regresion.csv ({len(df_bg):,} materiales)")
            else:
                print("Columna 'band_gap' no encontrada")
            
            # Caso 3: Estabilidad
            if 'energy_above_hull' in df.columns:
                df_est = df[df['energy_above_hull'].notna()].copy()
                df_est.to_csv("caso3_estabilidad.csv", index=False)
                print(f"caso3_estabilidad.csv ({len(df_est):,} materiales)")
            else:
                print("Columna 'energy_above_hull' no encontrada")
            
            # Mostrar estadísticas finales
            print("ESTADÍSTICAS FINALES DEL DATASET")
            print("=" * 50)
            print(f"Total materiales procesados: {len(df):,}")
            print(f"Total características por material: {len(df.columns)}")
            print(f"Errores durante procesamiento: {errores}")
            
            if 'is_semiconductor' in df.columns:
                semiconductores = df['is_semiconductor'].sum()
                print(f"Semiconductores (0.1 < bg < 4.0 eV): {semiconductores:,} ({semiconductores/len(df)*100:.1f}%)")
            if 'is_photovoltaic' in df.columns:
                fotovoltaicos = df['is_photovoltaic'].sum()
                print(f"Fotovoltaicos (0.8 ≤ bg ≤ 2.2 eV): {fotovoltaicos:,} ({fotovoltaicos/len(df)*100:.1f}%)")
            if 'is_stable' in df.columns:
                estables = df['is_stable'].sum()
                print(f"Estables (energy_above_hull < 0.1): {estables:,} ({estables/len(df)*100:.1f}%)")
            
            # Mostrar distribución de sistemas cristalinos
            if 'crystal_system' in df.columns:
                sistemas = df['crystal_system'].value_counts()
                print(f"Sistemas cristalinos más comunes:")
                for sistema, cantidad in sistemas.head(5).items():
                    if sistema:  # Solo mostrar sistemas no vacíos
                        print(f"  {sistema}: {cantidad:,} materiales")
            
            # Mostrar ejemplo de datos
            print("EJEMPLO DE PRIMEROS MATERIALES:")
            cols_ejemplo = ['material_id', 'formula', 'band_gap', 'crystal_system', 'is_semiconductor']
            cols_disponibles = [col for col in cols_ejemplo if col in df.columns]
            if cols_disponibles:
                print(df[cols_disponibles].head(3).to_string(index=False))
            
            print("¡DESCARGA DE {len(df):,} MATERIALES COMPLETADA EXITOSAMENTE!")
            print("Archivos CSV listos para tu proyecto de Machine Learning")
            
        else:
            print("No se procesaron materiales exitosamente")

if __name__ == "__main__":
    main()