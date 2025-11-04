export function Index() {
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
                        ></textarea>
                        <small
                            id="ayuda-noticia"
                            className="form-text text-muted d-block mt-2"
                        >
                            El texto se analizará para determinar si contiene
                            información falsa o verificada.
                        </small>
                    </div>

                    <button
                        type="submit"
                        className="btn btn-success btn-lg px-5"
                        aria-label="Analizar noticia"
                    >
                        Analizar
                    </button>
                </form>
            </main>
        </div>
    );
}
