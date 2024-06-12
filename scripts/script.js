let model;
const metadata = {
    labels: ["Crown Gal", "Bacterial Leaf Spot", "Fire Blight", "Bacterial Wilt", "Root Rot", "Late Blight", "Pythium Root Rot", "Mistletoe", "Lethal Yellowing", "Aster Yellows", "Blight", "Leaf Spot", "Rust", "Downy Mildew", "Powdery Milde", "Leaf Curl Virus", "Mosaic Virus"] // Replace with your actual class labels
};

// Information about each disease
const diseaseInfo = {
    "Crown Gal": "Caused by a bacterial infection that enters through wounds.<br>Avoid injuring plants and use sterilized tools.",
    "Bacterial Leaf Spot": "Caused by bacteria in warm, wet conditions.<br>Ensure good air circulation and avoid overhead watering.",
    "Fire Blight": "Bacterial disease spread by insects and rain.<br>Prune infected branches and disinfect tools between cuts.",
    "Bacterial Wilt": "Caused by soil-borne bacteria attacking the plant's vascular system.<br>Use resistant plant varieties and rotate crops.",
    "Root Rot": "Caused by overwatering and poor drainage.<br>Improve soil drainage and avoid overwatering plants.",
    "Late Blight": "Fungal disease favored by cool, moist conditions.<br>Use resistant varieties and apply fungicides preventatively.",
    "Pythium Root Rot": "Fungal disease caused by overly wet soils.<br>Ensure proper drainage and avoid waterlogged conditions.",
    "Mistletoe": "Parasitic plant that extracts nutrients from the host.<br>Remove mistletoe from infected branches promptly.",
    "Lethal Yellowing": "Caused by phytoplasma spread by insects.<br>Control insect vectors and remove infected plants to prevent spread.",
    "Aster Yellows": "Caused by phytoplasma and spread by leafhoppers.<br>Control leafhopper population and remove infected plants.",
    "Blight": "Caused by various fungal or bacterial pathogens.<br>Use resistant varieties and apply appropriate fungicides.",
    "Leaf Spot": "Fungal or bacterial infection causing spots on leaves.<br>Ensure good air circulation and avoid wetting foliage.",
    "Rust": "Fungal disease characterized by rust-colored pustules.<br>Remove affected leaves and apply fungicides if necessary.",
    "Downy Mildew": "Fungal disease favored by humid conditions.<br>Improve air circulation and apply fungicides preventatively.",
    "Powdery Milde": "Fungal disease that thrives in dry, warm conditions.<br>Ensure good air circulation and apply fungicides as needed.",
    "Leaf Curl Virus": "Caused by a virus transmitted by insects.<br>Control insect vectors and remove infected plants promptly.",
    "Mosaic Virus": "Viral infection spread by insect vectors.<br>Control insects and remove infected plants to prevent spread."
};

const loadModel = async () => {
    model = await tf.loadLayersModel('model/model.json');
    document.getElementById('prediction').innerText = "Upload an image to get a prediction.";
};

const loadImage = (event) => {
    const image = document.getElementById('input-image');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.onload = () => classifyImage(image);
};

const classifyImage = async (image) => {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();
    const predictions = await model.predict(tensor).data();
    const results = Array.from(predictions)
        .map((p, i) => ({ probability: p, className: metadata.labels[i] }))
        .sort((a, b) => b.probability - a.probability);

    const topResult = results[0];
    const diseaseName = topResult.className;

    document.getElementById('prediction').innerText = `Prediction: ${diseaseName}`;
    document.getElementById('disease-info').innerHTML = diseaseInfo[diseaseName];
};

window.onload = loadModel;
