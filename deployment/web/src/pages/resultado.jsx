import { useState } from "react";
import { PieChart } from "@mui/x-charts";
import { useNavigate } from "react-router-dom";

export function Resultado() {
    const navigate = useNavigate();
    const fake = 50;
    const real = 50;
    const [texto, setTexto] = useState(
        location.state?.newsText || "No se cargo la noticia correctamente..."
    );

    return (
        <div
            className="container-resultado d-flex justify-content-center align-items-center"
            role="region"
            aria-labelledby="titulo-principal"
        >
            <main className="main-resultado text-center" role="main">
                <h1
                    id="titulo-principal"
                    className="titulo-resultado mb-4 fw-bold"
                    tabIndex="0"
                >
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
                    <form
                        className="form-resultado"
                        aria-label="Formulario para analizar noticias"
                        onSubmit={(e) => e.preventDefault()}
                    >
                        <textarea
                            id="noticia"
                            className="form-control textarea-resultado mb-3"
                            rows="6"
                            placeholder="Pega aquí tu noticia en español..."
                            aria-describedby="ayuda-noticia"
                            aria-required="true"
                            value={texto}
                            onChange={(e) => setTexto(e.target.value)}
                        />

                        <select
                            className="form-select select-resultado mb-3"
                            aria-label="Seleccionar modelo de detección"
                            defaultValue="Hibrido"
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
                            className="btn btn-success w-100"
                            aria-label="Analizar noticia"
                        >
                            Analizar
                        </button>
                    </form>

                    {/* Columna 2: gráfico */}
                    <div
                        className="grafico-resultado"
                        role="img"
                        aria-label="Gráfico circular con distribución de probabilidades"
                    >
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

                        <p className="probabilidad-principal">
                            <strong>{fake}%</strong> de probabilidad de que la
                            noticia sea <strong>falsa</strong>.
                        </p>

                        <div className="resultado-item">
                            <span>🟦 Real:</span>
                            <span>{real}%</span>
                        </div>

                        <div className="resultado-item">
                            <span>🟥 Fake:</span>
                            <span>{fake}%</span>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
