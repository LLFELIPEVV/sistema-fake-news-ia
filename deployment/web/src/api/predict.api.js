import axios from "axios";

const api = axios.create({
    baseURL: "http://127.0.0.1:8000/api/v1",
});

// Función para enviar la noticia al backend y obtener la predicción
export const fetchPrediction = async (newsText, model) => {
    try {
        const payload = {
            text: newsText,
            model: model,
        };
        const response = await api.post("/predict", payload);
        console.log(response);
        console.log(response.data.confidence);
        return response.data;
    } catch (error) {
        console.error("Error en la predicción:", error);
        throw error;
    }
};
