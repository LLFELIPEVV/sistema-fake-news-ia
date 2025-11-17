import { PieChart } from "@mui/x-charts";
import { useState, useEffect } from "react";
import { fetchPrediction } from "../api/predict.api";
import { useNavigate, useLocation } from "react-router-dom";

export function Resultado() {
    const navigate = useNavigate();
    const location = useLocation();

    // Extraer datos iniciales
    const {
        result: initialResult,
        newsText: initialText,
        model: initialModel,
    } = location.state || {};

    // Estados locales
    const [texto, setTexto] = useState(initialText || "");
    const [modelo, setModelo] = useState(initialModel || "Hibrido");
    const [result, setResult] = useState(initialResult || null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Datos de la API
    const prediction = result?.prediction || "desconocido";
    const confidence = result?.confidence || 0;

    // Calcular porcentajes correctamente
    let fake = 50,
        real = 50;
    if (prediction === "fake") {
        fake = confidence * 100;
        real = (1 - confidence) * 100;
    } else if (prediction === "real") {
        real = confidence * 100;
        fake = (1 - confidence) * 100;
    }

    // Función de envío
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        if (texto.length < 40) {
            setError("La noticia debe tener al menos 40 caracteres.");
            setLoading(false);
            return;
        }

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

            const modeloReal = modelosMap[modelo];
            const data = await fetchPrediction(texto, modeloReal);
            setResult(data); // ✅ actualiza resultado sin redirigir
        } catch (err) {
            console.error(err);
            setError("Ocurrió un error al analizar la noticia.");
        } finally {
            setLoading(false);
        }
    };

    // En caso de acceso directo sin datos
    useEffect(() => {
        if (!initialResult && !initialText) navigate("/");
    }, [initialResult, initialText, navigate]);

    return (
        <div
            className="container-resultado d-flex justify-content-center align-items-center"
            role="region"
        >
            <main className="main-resultado text-center" role="main">
                <h1 className="titulo-resultado mb-4 fw-bold" tabIndex="0">
                    Detector de Fake News
                </h1>

                <button
                    className="btn-volver mb-4"
                    onClick={() => navigate("/")}
                >
                    ← Volver
                </button>

                <section className="grid-resultado">
                    {/* Columna 1: formulario */}
                    <form className="form-resultado" onSubmit={handleSubmit}>
                        <textarea
                            id="noticia"
                            className="form-control textarea-resultado mb-3"
                            rows="6"
                            placeholder="Pega aquí tu noticia en español..."
                            value={texto}
                            onChange={(e) => setTexto(e.target.value)}
                            required
                        />

                        <select
                            className="form-select select-resultado mb-3"
                            value={modelo}
                            onChange={(e) => setModelo(e.target.value)}
                        >
                            <option value="hibrido-mejor-fake">
                                Mejor detección de Fake News
                            </option>
                            <option value="rf-mejor-real">
                                Mejor detección de Real News
                            </option>
                            <option value="cnn-mejor-general">
                                Mejor modelo general
                            </option>
                            <option value="cnn-equilibrado">
                                Mejor en detección equilibrada
                            </option>
                            <option value="nb-rapido">
                                Mejor en rapidez y eficiencia
                            </option>
                            <option value="nb-estable">
                                Mejor en estabilidad y reproducibilidad
                            </option>
                            <option value="rf-sobreajuste">
                                Modelo con menor sobreajuste
                            </option>
                            <option value="cnn-consistente">
                                Mejor en consistencia general
                            </option>
                        </select>

                        <button
                            type="submit"
                            className="btn btn-success w-100"
                            disabled={loading}
                        >
                            {loading ? "Analizando..." : "Analizar"}
                        </button>

                        {error && <p className="text-danger mt-2">{error}</p>}
                    </form>

                    {/* Columna 2: gráfico */}
                    <div className="grafico-resultado">
                        <PieChart
                            colors={["#ff4d4d", "#4dabf7"]}
                            series={[
                                {
                                    data: [
                                        { id: 0, value: fake, label: "Fake" },
                                        { id: 1, value: real, label: "Real" },
                                    ],
                                },
                            ]}
                            width={250}
                            height={250}
                        />
                    </div>

                    {/* Columna 3: resultados */}
                    <div className="datos-resultado text-start">
                        <h2 className="fw-bold mb-3">
                            Resultados del análisis
                        </h2>

                        {result ? (
                            <>
                                <p className="probabilidad-principal">
                                    <strong>
                                        {prediction === "fake"
                                            ? fake.toFixed(1)
                                            : real.toFixed(1)}
                                        %
                                    </strong>{" "}
                                    de probabilidad de que la noticia sea{" "}
                                    <strong>
                                        {prediction === "fake"
                                            ? "falsa"
                                            : "real"}
                                    </strong>
                                    .
                                </p>

                                <div className="resultado-item">
                                    <span>🟦 Real:</span>{" "}
                                    <span>{real.toFixed(1)}%</span>
                                </div>

                                <div className="resultado-item">
                                    <span>🟥 Fake:</span>{" "}
                                    <span>{fake.toFixed(1)}%</span>
                                </div>
                            </>
                        ) : (
                            <p className="text-muted">
                                Analiza una noticia para ver los resultados.
                            </p>
                        )}
                    </div>
                </section>
            </main>
        </div>
    );
}
