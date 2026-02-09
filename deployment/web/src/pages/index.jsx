import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchPrediction } from "../api/predict.api";

export function Index() {
    const navigate = useNavigate();
    const [newsText, setNewsText] = useState("");
    const [model, setModel] = useState("hibrido-mejor-fake");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        if (newsText.length > 40 && model) {
            try {
                const modelosMap = {
                    "hibrido-mejor-fake": "Hibrido",
                    "rf-mejor-real": "Random Forest",
                    "cnn-mejor-general": "CNN",
                    "cnn-equilibrado": "CNN",
                    "nb-rapido": "Naive Bayes",
                    "nb-estable": "Naive Bayes",
                    "rf-sobreajuste": "Random Forest",
                    "cnn-consistente": "CNN",
                };

                const modeloReal = modelosMap[model];
                const data = await fetchPrediction(newsText, modeloReal);

                navigate("/resultado", {
                    state: { result: data, newsText, model },
                });
            } catch (err) {
                console.error(err);
                setError("Ocurrió un error durante el análisis del texto.");
            } finally {
                setLoading(false);
            }
        } else {
            setError("La noticia debe tener al menos 40 caracteres.");
            setLoading(false);
        }
    };

    return (
        <div
            className="container-index d-flex justify-content-center align-items-center"
            role="region"
            aria-labelledby="titulo-principal"
        >
            <main className="main-index text-center" role="main">
                <h1
                    id="titulo-principal"
                    className="titulo-index mb-4 fw-bold"
                    tabIndex="0"
                >
                    Analizador de noticias basado en PLN
                </h1>

                <form
                    className="w-100"
                    aria-label="Formulario para el análisis lingüístico de noticias"
                    onSubmit={handleSubmit}
                >
                    <div className="mb-3">
                        <label htmlFor="noticia" className="visually-hidden">
                            Texto de la noticia
                        </label>

                        <textarea
                            id="noticia"
                            className="form-control textarea-index"
                            rows="5"
                            placeholder="Pega aquí una noticia en español para su análisis..."
                            aria-describedby="ayuda-noticia"
                            aria-required="true"
                            value={newsText}
                            onChange={(e) => setNewsText(e.target.value)}
                        ></textarea>

                        <small
                            id="ayuda-noticia"
                            className="form-text text-muted d-block mt-2"
                        >
                            El sistema analiza el texto ingresado utilizando
                            procesamiento de lenguaje natural para estimar su
                            similitud con patrones lingüísticos asociados a
                            noticias falsas y reales. No verifica hechos ni
                            contrasta fuentes externas.
                        </small>
                    </div>

                    <select
                        className="form-select select-index mb-4"
                        aria-label="Seleccionar modelo de análisis"
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                    >
                        <option value="hibrido-mejor-fake">
                            Mejor detección de patrones Fake
                        </option>
                        <option value="rf-mejor-real">
                            Mejor detección de patrones Reales
                        </option>
                        <option value="cnn-mejor-general">
                            Mejor modelo general
                        </option>
                        <option value="cnn-equilibrado">
                            Mejor equilibrio entre clases
                        </option>
                        <option value="nb-rapido">
                            Mayor rapidez y eficiencia
                        </option>
                        <option value="nb-estable">
                            Mayor estabilidad y reproducibilidad
                        </option>
                        <option value="rf-sobreajuste">
                            Menor sobreajuste
                        </option>
                        <option value="cnn-consistente">
                            Mayor consistencia general
                        </option>
                    </select>

                    <button
                        type="submit"
                        className="btn btn-success btn-lg px-5"
                        aria-label="Iniciar análisis del texto"
                        disabled={loading}
                    >
                        {loading ? "Analizando texto..." : "Analizar"}
                    </button>
                </form>

                {error && <p className="text-danger mt-3">{error}</p>}
            </main>
        </div>
    );
}
