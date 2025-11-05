import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Index } from "./pages/index";
import { Resultado } from "./pages/resultado";

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/resultado" element={<Resultado />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
