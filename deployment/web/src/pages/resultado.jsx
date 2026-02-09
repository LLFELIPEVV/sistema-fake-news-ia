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
    const [modelo, setModelo] = useState(initialModel || "hibrido-mejor-fake");
    const [result, setResult] = useState(initialResult || null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Datos de la API
    const prediction = result?.prediction || "desconocido";
    const confidence = result?.confidence || 0;

    // Calcular porcentajes (probabilidad de clase)
    let fake = 50,
        real = 50;

    if (prediction === "fake") {
        fake = confidence * 100;
        real = (1 - confidence) * 100;
    } else if (prediction === "real") {
        real = confidence * 100;
        fake = (1 - confidence) * 100;
    }

    // Envío del formulario
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        if (texto.length < 40) {
            setError("El texto debe tener al menos 40 caracteres.");
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
            setResult(data);
        } catch (err) {
            console.error(err);
            setError("Ocurrió un error al analizar la noticia.");
        } finally {
            setLoading(false);
        }
    };

    // Evitar acceso directo sin datos
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
                    Analizador de noticias basado en PLN
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
                            placeholder="Pega aquí una noticia en español..."
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
                                        {
                                            id: 0,
                                            value: fake,
                                            label: "Patrones de noticias falsas",
                                        },
                                        {
                                            id: 1,
                                            value: real,
                                            label: "Patrones de noticias reales",
                                        },
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
                            Resultados del análisis lingüístico
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
                                    de mayor coincidencia con patrones de
                                    noticias{" "}
                                    <strong>
                                        {prediction === "fake"
                                            ? "potencialmente falsas"
                                            : "potencialmente reales"}
                                    </strong>
                                    .
                                </p>

                                <div className="resultado-item">
                                    <span>
                                        🟦 Coincidencia con patrones reales:
                                    </span>{" "}
                                    <span>{real.toFixed(1)}%</span>
                                </div>

                                <div className="resultado-item">
                                    <span>
                                        🟥 Coincidencia con patrones falsos:
                                    </span>{" "}
                                    <span>{fake.toFixed(1)}%</span>
                                </div>

                                <p className="text-muted small mt-2 mb-0">
                                    Este resultado es orientativo y se basa solo
                                    en el análisis del texto. El modelo no
                                    verifica hechos ni fuentes externas. La
                                    decisión final no depende únicamente del
                                    porcentaje más alto, sino del peso que el
                                    modelo asigna a distintos patrones
                                    aprendidos, lo que puede hacer que una
                                    noticia se clasifique como falsa aunque
                                    tenga mayor coincidencia con patrones
                                    reales.
                                </p>
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
