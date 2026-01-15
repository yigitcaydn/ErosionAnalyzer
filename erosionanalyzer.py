import tkinter
import customtkinter as ctk
import threading
from tkinter import messagebox
import ee
import os
import requests
import rasterio
from rasterio.plot import plotting_extent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import regionprops
from scipy.ndimage import label, binary_dilation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- 1. GOOGLE EARTH ENGINE ---
try:
    ee.Initialize(project='ee-yigitdgp')
except:
    ee.Authenticate()
    ee.Initialize(project='ee-yigitdgp')

# --- 2. AYARLAR ---
def get_satellite_config(year):
    if year >= 2016:
        return {
            "type": "SENTINEL", 
            "collection": "COPERNICUS/S2_SR_HARMONIZED", 
            "bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B8", "SWIR": "B11"}, 
            "vis_min": 0, "vis_max": 3000
        }
    else:
        collection = "LANDSAT/LC08/C02/T1_L2" if year >= 2013 else "LANDSAT/LT05/C02/T1_L2"
        bands = {"R": "SR_B4", "G": "SR_B3", "B": "SR_B2", "NIR": "SR_B5", "SWIR": "SR_B6"} if year >= 2013 else \
                {"R": "SR_B3", "G": "SR_B2", "B": "SR_B1", "NIR": "SR_B4", "SWIR": "SR_B5"}
        return {
            "type": "LANDSAT", "collection": collection, "bands": bands, 
            "vis_min": 0, "vis_max": 30000
        }

def safe_download(image_obj, region, filepath, log_callback):
    safe_image = image_obj.unmask(0)
    params = {'scale': 30, 'crs': 'EPSG:4326', 'region': region, 'format': 'GEO_TIFF'}

    try:
        url = safe_image.getDownloadURL(params)
        response = requests.get(url)

        if response.status_code == 400:
            log_callback("⚠️ 30m ağır geldi, 100m deneniyor...")
            params['scale'] = 100
            url = safe_image.getDownloadURL(params)
            response = requests.get(url)

        if response.status_code == 200:
            with open(filepath, 'wb') as f: f.write(response.content)
            log_callback(f"✅ İndirildi: {os.path.basename(filepath)}")
            return True
        else:
            log_callback(f"❌ İndirme Başarısız (Kod: {response.status_code})")
            return False
    except Exception as e:
        log_callback(f"❌ Bağlantı Hatası: {e}")
        return False

# --- 3. ANALİZ MOTORU (HİBRİT + RANDOM FOREST) ---
def analiz_motoru(koordinat, start_year, end_year, log_callback):
    roi = ee.Geometry.Rectangle(koordinat)
    dosya_veri = "Analiz_Data_GrandMaster.tif"
    dosya_img = "Analiz_Img_GrandMaster.tif"
    
    if os.path.exists(dosya_veri): os.remove(dosya_veri)
    if os.path.exists(dosya_img): os.remove(dosya_img)

    sat_cfg_start = get_satellite_config(start_year)
    sat_cfg_end = get_satellite_config(end_year)

    def execute_logic(method_type):
        bands_s = sat_cfg_start["bands"]
        bands_e = sat_cfg_end["bands"]

        col_start = ee.ImageCollection(sat_cfg_start["collection"]).filterBounds(roi).filterDate(f'{start_year}-01-01', f'{start_year}-12-31')
        col_end = ee.ImageCollection(sat_cfg_end["collection"]).filterBounds(roi).filterDate(f'{end_year}-01-01', f'{end_year}-12-31')

        if method_type == 'median':
            log_callback("⚙️ Yöntem: MEDIAN (Yüksek Kalite)...")
            img_start = col_start.median().clip(roi)
            img_end = col_end.median().clip(roi)
        else:
            log_callback("⚙️ Yöntem: FAST (Hızlı Mod)...")
            img_start = col_start.sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(roi)
            img_end = col_end.sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(roi)

        # --- A. KLASİK RİSK HESAPLAMALARI ---
        ndvi = img_start.normalizedDifference([bands_s["NIR"], bands_s["R"]]).rename('ndvi').unmask(0)
        risk_ndvi = ndvi.multiply(-1).add(1).clamp(0, 1)

        soil_sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').clip(roi).rename('sand').unmask(0)
        risk_sand = soil_sand.divide(100).clamp(0, 1)

        soil_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').clip(roi).rename('clay').unmask(0) 
        risk_clay = soil_clay.divide(100).multiply(-1).add(1).clamp(0, 1)

        dem = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select('DSM').mosaic()
        slope = ee.Terrain.slope(dem).clip(roi).rename('slope').unmask(0)
        risk_slope = slope.divide(30).clamp(0, 1)

        mndwi = img_start.normalizedDifference([bands_s["G"], bands_s["SWIR"]]).rename('mndwi')

        base_risk = risk_sand.multiply(0.40).add(risk_ndvi.multiply(0.30)).add(risk_clay.multiply(0.15)).add(risk_slope.multiply(0.15))
        land_mask = mndwi.lt(0.2) 
        total_risk = base_risk.multiply(land_mask).multiply(100).toInt8().rename('risk')

        # --- B. EROZYON TESPİTİ ---
        is_land_start = img_start.normalizedDifference([bands_s["G"], bands_s["SWIR"]]).lt(0.2).And(img_start.select(bands_s["NIR"]).gt(500))
        is_water_end = img_end.normalizedDifference([bands_e["G"], bands_e["SWIR"]]).gt(0.1)
        
        raw_erosion = is_water_end.And(is_land_start)
        filtered_erosion = raw_erosion.multiply(soil_sand.gt(1)).rename('erosion').toInt8()

        # --- C. RANDOM FOREST---
        log_callback("🤖 AI Model: Random Forest Eğitiliyor...")
        
        features_img = ndvi.addBands([mndwi, soil_sand, soil_clay, slope]).unmask(0)
        training_input = features_img.addBands(filtered_erosion)
        
        training_samples = training_input.stratifiedSample(
            numPoints=1000, 
            classBand='erosion', 
            region=roi, 
            scale=30, 
            seed=42,
            tileScale=4 
        )

        classifier = ee.Classifier.smileRandomForest(numberOfTrees=20).train(
            features=training_samples, 
            classProperty='erosion', 
            inputProperties=['ndvi', 'mndwi', 'sand', 'clay', 'slope']
        )

        ml_prediction = features_img.classify(classifier).rename('ml_risk').toInt8()

        # --- D. ÇIKTI BİRLEŞTİRME ---
        final_data = total_risk.addBands(filtered_erosion).addBands(ml_prediction)
        
        vis_bands = [bands_s["R"], bands_s["G"], bands_s["B"]]
        visual_rgb = img_start.select(vis_bands).visualize(min=sat_cfg_start["vis_min"], max=sat_cfg_start["vis_max"])

        return final_data, visual_rgb

    # İşlem Denemesi
    try:
        data_to_dl, img_to_dl = execute_logic('median')
        if safe_download(data_to_dl, roi, dosya_veri, log_callback):
            if safe_download(img_to_dl, roi, dosya_img, log_callback):
                return (dosya_veri, dosya_img)
    except Exception as e:
        log_callback(f"⚠️ Median Hata: {e}")
        log_callback("🔄 Hızlı Mod deneniyor...")
        try:
            data_to_dl, img_to_dl = execute_logic('first')
            if safe_download(data_to_dl, roi, dosya_veri, log_callback):
                if safe_download(img_to_dl, roi, dosya_img, log_callback):
                    return (dosya_veri, dosya_img)
        except Exception as e2:
            log_callback(f"❌ Kritik Hata: {e2}")

    return (None, None)

# --- 4. GÖRSELLEŞTİRME (GUI) ---
def open_plot_window(dosya_veri, dosya_img):
    try:
        with rasterio.open(dosya_img) as src_img:
            img_data = np.moveaxis(src_img.read(), 0, -1)
            extent = plotting_extent(src_img)
            img_data = img_data.astype(float)
            if img_data.max() > 0:
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

        with rasterio.open(dosya_veri) as src_data:
            risk_map = src_data.read(1, out_shape=(src_img.height, src_img.width))    # Band 1: MCDA
            erosion_map = src_data.read(2, out_shape=(src_img.height, src_img.width)) # Band 2: Erozyon
            ml_map = src_data.read(3, out_shape=(src_img.height, src_img.width))      # Band 3: ML
    except Exception as e:
        print(f"Veri okuma hatası: {e}")
        return

    plot_window = tkinter.Toplevel()
    plot_window.title("Analiz Sonuçları - (MCDA + Profiling + ML)")
    plot_window.geometry("1200x750")
    plot_window.configure(bg="#2b2b2b")

    map_frame = tkinter.Frame(plot_window, bg="black")
    map_frame.pack(side="left", fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#2b2b2b")
    ax.axis('off')
    
    img_layer = ax.imshow(img_data, extent=extent)
    
    erosion_visual = np.zeros((erosion_map.shape[0], erosion_map.shape[1], 4))
    erosion_visual[erosion_map == 1] = [1, 0, 0, 1] 
    erosion_layer = ax.imshow(erosion_visual, extent=extent, zorder=2)

    interactive_elements = []

    # 1. YEŞİL KUTULAR (MCDA)
    erosion_grouped = binary_dilation(erosion_map == 1, iterations=1)
    label_img, _ = label(erosion_grouped)
    
    for props in regionprops(label_img):
        if props.area < 5: continue
        min_row, min_col, max_row, max_col = props.bbox
        if (max_col - min_col) > (risk_map.shape[1] * 0.4): continue 

        with rasterio.open(dosya_img) as src:
            left, top = src.xy(min_row, min_col)
            right, bottom = src.xy(max_row, max_col)
            width = right - left
            height = top - bottom
            risk_slice = risk_map[min_row:max_row, min_col:max_col]
            risk_values = risk_slice[risk_slice > 0]
            avg_risk = int(np.mean(risk_values)) if len(risk_values) > 0 else 0
            
            if avg_risk > 20:
                rect = patches.Rectangle((left, bottom), width, height, linewidth=1.5, edgecolor='#00FF00', facecolor='none', zorder=4)
                ax.add_patch(rect)
                txt = ax.text(left, top, f"R:%{avg_risk}", color='white', fontsize=7, fontweight='bold', bbox=dict(facecolor='#00FF00', alpha=0.6, edgecolor='none'), zorder=5)
                
                interactive_elements.append({
                    'rect': rect, 'text': txt, 'type': 'green', 'orig_color': '#00FF00', 'orig_text': f"R:%{avg_risk}", 'active': True
                })

    # 2. TURUNCU KUTULAR  
    eroded_risk_values = risk_map[erosion_map == 1]
    if len(eroded_risk_values) > 0:
        learned_mean = np.mean(eroded_risk_values)
        dynamic_threshold = learned_mean * 0.85 # %15 Tolerans
        print(f"Sistem Öğrendi: Kritik Eşik {int(dynamic_threshold)}")
    else:
        dynamic_threshold = 40 # Varsayılan

    # --- B. BENZERLİKTARAMASI ---
    profile_risk = (risk_map > dynamic_threshold) & (erosion_map == 0)
    proximity_risk = binary_dilation(erosion_map == 1, iterations=5) & (erosion_map == 0)
    
    future_risk_mask = profile_risk | proximity_risk
    
    # --- C. GÖRSELLEŞTİRME ---
    label_future, _ = label(future_risk_mask) 
    regions_future = regionprops(label_future)

    for props in regions_future:
        if props.area < 50: continue 
        min_row, min_col, max_row, max_col = props.bbox
        if (max_col - min_col) > (risk_map.shape[1] * 0.2): continue

        with rasterio.open(dosya_img) as src:
            left, top = src.xy(min_row, min_col)
            right, bottom = src.xy(max_row, max_col)
            width = right - left
            height = top - bottom
            risk_slice = risk_map[min_row:max_row, min_col:max_col]
            valid_risks = risk_slice[risk_slice > 0]
            avg_future_risk = int(np.mean(valid_risks)) if len(valid_risks) > 0 else 0

            rect = patches.Rectangle((left, bottom), width, height, linewidth=1.5, edgecolor='#FFA500', linestyle='--', facecolor='none', zorder=4)
            ax.add_patch(rect)
            txt = ax.text(left, top, f"Tahmin: %{avg_future_risk}", color='black', fontsize=8, fontweight='bold', bbox=dict(facecolor='#FFA500', alpha=0.8, edgecolor='none'), zorder=5)
            
            interactive_elements.append({
                'rect': rect, 'text': txt, 'type': 'orange', 'orig_color': '#FFA500', 'orig_text': f"Tahmin: %{avg_future_risk}", 'active': True
            })

    ax.set_title("Analiz Sonuçları (Akıllı Profilleme Aktif)", color="white", fontsize=14)

    # 3. MOR KUTULAR (AI - Random Forest)
    ml_mask = (ml_map == 1) & (erosion_map == 0)
    label_ml, _ = label(ml_mask) 
    
    for props in regionprops(label_ml):
        if props.area < 30: continue
        min_row, min_col, max_row, max_col = props.bbox
        
        with rasterio.open(dosya_img) as src:
            left, top = src.xy(min_row, min_col)
            right, bottom = src.xy(max_row, max_col)
            width = right - left
            height = top - bottom
            
            rect = patches.Rectangle((left, bottom), width, height, linewidth=2, edgecolor='#D000FF', linestyle=':', facecolor='none', zorder=6)
            ax.add_patch(rect)
            txt = ax.text(left, bottom, "AI", color='white', fontsize=7, fontweight='bold', bbox=dict(facecolor='#D000FF', alpha=0.8, edgecolor='none'), zorder=7)
            
            interactive_elements.append({
                'rect': rect, 'text': txt, 'type': 'purple', 'orig_color': '#D000FF', 'orig_text': "AI", 'active': True
            })

    ax.set_title("YEŞİL: Bilimsel Risk | TURUNCU: İstatistiksel Profil | MOR: Yapay Zeka (RF)", color="white", fontsize=11)

    canvas = FigureCanvasTkAgg(fig, master=map_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def on_click(event):
        if event.inaxes != ax: return 
        for item in interactive_elements:
            rect = item['rect']
            contains, _ = rect.contains(event)
            if contains:
                item['active'] = not item['active']
                if not item['active']:
                    rect.set_edgecolor('gray')
                    rect.set_alpha(0.3)
                    item['text'].set_visible(False)
                else:
                    rect.set_edgecolor(item['orig_color'])
                    rect.set_alpha(1.0)
                    item['text'].set_visible(True)
                canvas.draw()
                break
    
    canvas.mpl_connect('button_press_event', on_click)
    toolbar = NavigationToolbar2Tk(canvas, map_frame)
    toolbar.update()

    # KONTROL PANELİ
    control_panel = ctk.CTkFrame(plot_window, width=250, corner_radius=0)
    control_panel.pack(side="right", fill="y")
    lbl_control = ctk.CTkLabel(control_panel, text="KATMAN YÖNETİMİ", font=("Roboto", 16, "bold"))
    lbl_control.pack(pady=(20, 20), padx=20)

    def toggle_layer(layer_type):
        val = switches[layer_type].get()
        if layer_type == 'sat': img_layer.set_visible(val)
        elif layer_type == 'ero': erosion_layer.set_visible(val)
        else:
            for item in interactive_elements:
                if item['type'] == layer_type:
                    item['rect'].set_visible(val)
                    item['text'].set_visible(val)
        canvas.draw()

    switches = {}
    
    switches['sat'] = ctk.CTkSwitch(control_panel, text="Uydu Görüntüsü", command=lambda: toggle_layer('sat'))
    switches['sat'].select(); switches['sat'].pack(pady=10, padx=20, anchor="w")
    
    switches['ero'] = ctk.CTkSwitch(control_panel, text="Gerçek Erozyon (Kırmızı)", command=lambda: toggle_layer('ero'), progress_color="red")
    switches['ero'].select(); switches['ero'].pack(pady=10, padx=20, anchor="w")
    
    switches['green'] = ctk.CTkSwitch(control_panel, text="Bilimsel Risk (Yeşil)", command=lambda: toggle_layer('green'), progress_color="green")
    switches['green'].select(); switches['green'].pack(pady=10, padx=20, anchor="w")
    
    switches['orange'] = ctk.CTkSwitch(control_panel, text="Profil Analizi (Turuncu)", command=lambda: toggle_layer('orange'), progress_color="orange")
    switches['orange'].select(); switches['orange'].pack(pady=10, padx=20, anchor="w")

    switches['purple'] = ctk.CTkSwitch(control_panel, text="Yapay Zeka - RF (Mor)", command=lambda: toggle_layer('purple'), progress_color="#D000FF")
    switches['purple'].select(); switches['purple'].pack(pady=10, padx=20, anchor="w")

    info_box = ctk.CTkTextbox(control_panel, height=250, fg_color="#333333", text_color="white")
    info_box.pack(pady=30, padx=10, fill="x")
    info_msg = (
        "!!! HİBRİT ANALİZ RAPORU !!!\n\n"
        "1️⃣ YEŞİL (MCDA): Toprak, eğim ve bitki örtüsüne göre fiziksel risk alanları.\n\n"
        "2️⃣ TURUNCU (PROFILING - MCDA): Mevcut erozyonun karakteristiğinin benzer olduğu alanlar.\n\n"
        "3️⃣ MOR (AI - RANDOM FOREST): Random Forest Tahmini.\n\n"
        "Turuncu ve Mor kutuların kesiştiği yerler TAHMİNİ YÜKSEK RİSK bölgesidir."
    )
    info_box.insert("0.0", info_msg)
    info_box.configure(state="disabled")

# --- 5. ANA GUI ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ErozyonApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Erosion RiskAnalyzer")
        self.geometry("450x700")

        self.label_title = ctk.CTkLabel(self, text="KIYI EROZYONU ANALİZİ", font=("Roboto", 20, "bold"))
        self.label_title.pack(pady=20)

        self.frame_region = ctk.CTkFrame(self)
        self.frame_region.pack(pady=10, padx=20, fill="x")
        self.label_region = ctk.CTkLabel(self.frame_region, text="Bölge Seçimi:", font=("Roboto", 14))
        self.label_region.pack(anchor="w", padx=10, pady=5)

        self.regions = {
            "Sakarya Karasu": "30.63, 41.10, 30.72, 41.14",
            "Gediz Deltası": "26.78, 38.50, 26.95, 38.62",
            "Özel Koordinat": ""
        }
        self.option_region = ctk.CTkOptionMenu(self.frame_region, values=list(self.regions.keys()), command=self.update_coords)
        self.option_region.set("Sakarya Karasu")
        self.option_region.pack(fill="x", padx=10, pady=5)

        self.entry_coords = ctk.CTkEntry(self.frame_region, placeholder_text="Örn: 30.63, 41.10, 30.72, 41.14")
        self.entry_coords.pack(fill="x", padx=10, pady=5)
        self.entry_coords.insert(0, self.regions["Sakarya Karasu"])

        self.frame_years = ctk.CTkFrame(self)
        self.frame_years.pack(pady=10, padx=20, fill="x")
        self.label_years = ctk.CTkLabel(self.frame_years, text="Analiz Yılı Aralığı:", font=("Roboto", 14))
        self.label_years.pack(anchor="w", padx=10, pady=5)
        self.entry_start = ctk.CTkEntry(self.frame_years, placeholder_text="Başlangıç")
        self.entry_start.pack(side="left", padx=10, pady=10, expand=True)
        self.entry_start.insert(0, "2008")
        self.entry_end = ctk.CTkEntry(self.frame_years, placeholder_text="Bitiş")
        self.entry_end.pack(side="right", padx=10, pady=10, expand=True)
        self.entry_end.insert(0, "2014")

        self.btn_run = ctk.CTkButton(self, text="ANALİZİ BAŞLAT (AI)", font=("Roboto", 14, "bold"), height=40, fg_color="#8A2BE2", hover_color="#7A1ED2", command=self.start_thread)
        self.btn_run.pack(pady=20, padx=20, fill="x")

        self.textbox_log = ctk.CTkTextbox(self, height=150)
        self.textbox_log.pack(pady=10, padx=20, fill="both", expand=True)
        self.textbox_log.insert("0.0", "Sistem hazır. Random Forest + Orijinal Profilleme\n")

    def update_coords(self, choice):
        coords = self.regions[choice]
        self.entry_coords.delete(0, "end")
        self.entry_coords.insert(0, coords)

    def log(self, message):
        self.textbox_log.insert("end", message + "\n")
        self.textbox_log.see("end")

    def start_thread(self):
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def run_analysis(self):
        self.btn_run.configure(state="disabled", text="Yapay Zeka Çalışıyor...")
        self.textbox_log.delete("1.0", "end")
        try:
            coord_str = self.entry_coords.get()
            coords = [float(x.strip()) for x in coord_str.split(',')]
            s_year = int(self.entry_start.get())
            e_year = int(self.entry_end.get())
            self.log(f"📍 Bölge: {coords}")
            self.log(f"🤖 AI Modu: (Random Forest)")
            
            f_data, f_img = analiz_motoru(coords, s_year, e_year, self.log)
            
            if f_data and f_img:
                self.log("✅ Analiz tamamlandı! Hibrit harita açılıyor...")
                self.after(0, lambda: open_plot_window(f_data, f_img))
            else:
                self.log("❌ İşlem başarısız oldu.")
        except Exception as e:
            self.log(f"❌ HATA: {str(e)}")
        finally:
            self.btn_run.configure(state="normal", text="ANALİZİ BAŞLAT")

if __name__ == "__main__":
    app = ErozyonApp()
    app.mainloop()
