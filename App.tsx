import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HealthProvider } from './context/HealthContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import ProfilePage from './pages/ProfilePage';
import SymptomsPage from './pages/SymptomsPage';
import ResultsPage from './pages/ResultsPage';
import About from './pages/About';
import Contact from './pages/Contact';

function App() {
  return (
    <BrowserRouter>
      <HealthProvider>
        <div className="min-h-screen">
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/symptoms" element={<SymptomsPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
        </div>
      </HealthProvider>
    </BrowserRouter>
  );
}

export default App;
