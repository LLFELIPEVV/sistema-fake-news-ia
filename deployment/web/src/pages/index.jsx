import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchPrediction } from "../api/predict.api";

export function Index() {
    const navigate = useNavigate();
    const [newsText, setNewsText] = useState("");
    const [model, setModel] = useState("Hibrido");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        if (newsText.length > 40 && model) {
            try {
                const data = await fetchPrediction(newsText, model);
                navigate("/resultado", {
                    state: { result: data, newsText, model },
                });
            } catch (err) {
                console.error(err);
                setError("Ocurrió un error al analizar la noticia.");
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
                    Detector de Fake News
                </h1>

                <form
                    className="w-100"
                    aria-label="Formulario para analizar noticias"
                    onSubmit={handleSubmit}
                >
                    <div className="mb-4">
                        <label htmlFor="noticia" className="visually-hidden">
                            Pega tu noticia aquí
                        </label>
                        <textarea
                            id="noticia"
                            className="form-control textarea-index"
                            rows="5"
                            placeholder="Pega aquí tu noticia en español..."
                            aria-describedby="ayuda-noticia"
                            aria-required="true"
                            value={newsText}
                            onChange={(e) => setNewsText(e.target.value)}
                        ></textarea>
                        <small
                            id="ayuda-noticia"
                            className="form-text text-muted d-block mt-2"
                        >
                            El texto se analizará para determinar si contiene
                            información falsa o verificada.
                        </small>
                    </div>

                    <select
                        className="form-select select-index mb-4"
                        aria-label="Seleccionar modelo de detección"
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                    >
                        <option value="Hibrido">
                            Mejor detección de Fake News
                        </option>
                        <option value="Random Forest">
                            Mejor detección de Real News
                        </option>
                        <option value="CNN">Mejor modelo general</option>
                        <option value="CNN">
                            Mejor en detección equilibrada de ambos tipos de
                            noticias
                        </option>
                        <option value="Naive Bayes">
                            Mejor en rapidez y eficiencia
                        </option>
                        <option value="Naive Bayes">
                            Mejor en estabilidad y reproducibilidad
                        </option>
                        <option value="Random Forest">
                            Modelo con menor sobreajuste
                        </option>
                        <option value="CNN">
                            Mejor en consistencia general
                        </option>
                    </select>

                    <button
                        type="submit"
                        className="btn btn-success btn-lg px-5"
                        aria-label="Analizar noticia"
                        disabled={loading}
                    >
                        {loading ? "Analizando..." : "Analizar"}
                    </button>
                </form>

                {error && <p className="text-danger mt-3">{error}</p>}
            </main>
        </div>
    );
}
