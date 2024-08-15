import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import nibabel as nib
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the UNetGenerator architecture directly in the app.py
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(self._block(feature * 2, feature))
        
        # Final Convolution Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        enc_skips = []
        
        for layer in self.encoder:
            x = layer(x)
            enc_skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        enc_skips = enc_skips[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            enc_skip = enc_skips[idx//2]
            
            if x.shape != enc_skip.shape:
                x = transforms.functional.resize(x, enc_skip.shape[2:])
            
            concat_skip = torch.cat((enc_skip, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        return self.final_conv(x)

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/generated/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load('model/unet_generator.pth', map_location=device))
generator.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_nifti_image(nifti_path):
    img = nib.load(nifti_path)
    img_data = img.get_fdata()
    img_2d = img_data[:, :, img_data.shape[2] // 2]  # Taking the middle slice
    img_2d = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d))
    img_2d = Image.fromarray(np.uint8(img_2d * 255))
    img_2d = transform(img_2d)
    return img_2d.unsqueeze(0)  # Add batch dimension

def save_generated_image(tensor, output_path):
    img_np = tensor.squeeze().cpu().detach().numpy()
    img_pil = Image.fromarray(np.uint8(img_np * 255), mode='L')
    img_pil.save(output_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            input_img = load_nifti_image(filepath).to(device)
            output_img = generator(input_img)
            
            output_filename = 'generated.png'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            save_generated_image(output_img, output_path)
            
            return redirect(url_for('display_image', filename=output_filename))
    
    return render_template('index.html')

@app.route('/generated/<filename>')
def display_image(filename):
    return render_template('display.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
