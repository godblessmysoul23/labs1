import os
from flask import Flask, render_template, request, redirect
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def make_hist(img_np, filename):
    plt.figure(figsize=(4, 2.5))
    for i, col in enumerate(('r', 'g', 'b')):
        hist, _ = np.histogram(img_np[:, :, i], bins=256, range=(0, 256))
        plt.plot(hist, color=col)
    plt.xlim([0, 256])
    plt.tight_layout()
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(path)
    plt.close()
    return path

@app.route('/', methods=['GET', 'POST'])
def index():
    #Если файл загружен
    if request.method == 'POST': 
        file = request.files.get('file')
        try:
            period = float(request.form.get('period', 2.0))
        except:
            period = 2.0
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)#Сохраняем оригинальный файл

            #Наложение синуса по осям X и Y(обработка изображения)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img, dtype=np.float32)
            h, w, c = img_np.shape
            X, Y = np.meshgrid(np.linspace(0, 2*np.pi*period, w), np.linspace(0, 2*np.pi*period, h))
            func = np.expand_dims((np.sin(X + Y) + 1.0) / 2.0, axis=2)
            res_np = np.clip(img_np * func, 0, 255).astype(np.uint8)

            #Сохраняем обработнанное изображение
            res_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mod_' + file.filename)
            Image.fromarray(res_np).save(res_path)

            #Создаем гистрограммы для оригинального и обработанного файла
            h_orig = make_hist(img_np.astype(np.uint8), 'h_orig_' + file.filename + '.png')
            h_res = make_hist(res_np, 'h_res_' + file.filename + '.png')

            #Отображаем результат
            return render_template('result.html', orig=img_path, res=res_path, ho=h_orig, hr=h_res)
        
    #Иначе возвращаем на главную страницу    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)