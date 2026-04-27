import React, { useState, useCallback, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { AnimatePresence, motion } from "framer-motion";
import { Globe as GlobeIcon, Lightning, CaretDown } from "@phosphor-icons/react";
import GlobeView from "./components/GlobeView";
import CountryPanel from "./components/CountryPanel";
import NERResults from "./components/NERResults";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [countryText, setCountryText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [countries, setCountries] = useState([]);

  useEffect(() => {
    axios.get(`${API}/countries`).then(res => {
      setCountries(res.data.countries || []);
    }).catch(() => {});
  }, []);

  const handleCountrySelect = useCallback(async (countryName) => {
    setSelectedCountry(countryName);
    setResults(null);
    setShowResults(false);
    try {
      const res = await axios.get(
        `${API}/countries/${encodeURIComponent(countryName)}/context`
      );
      setCountryText(res.data.text);
    } catch {
      setCountryText(
        `Enter text about ${countryName} for NER analysis...`
      );
    }
  }, []);

  const handleClosePanel = useCallback(() => {
    setSelectedCountry(null);
    setCountryText("");
    setResults(null);
    setShowResults(false);
  }, []);

  const handleAnalyze = useCallback(
    async (text, modelType) => {
      setIsLoading(true);
      try {
        const res = await axios.post(`${API}/ner/analyze`, {
          text,
          model_type: modelType,
          country: selectedCountry,
        });
        setResults(res.data);
        setShowResults(true);
      } catch (err) {
        console.error("NER analysis failed:", err);
      } finally {
        setIsLoading(false);
      }
    },
    [selectedCountry]
  );

  const handleCloseResults = useCallback(() => {
    setShowResults(false);
  }, []);

  return (
    <div data-testid="app-root" className="relative w-screen h-screen overflow-hidden bg-[#0A0A0A]">
      {/* 3D Globe Background */}
      <GlobeView
        onCountrySelect={handleCountrySelect}
        selectedCountry={selectedCountry}
      />

      {/* Top-left branding */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="absolute top-6 left-6 z-10"
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-white/10 border border-white/10 flex items-center justify-center backdrop-blur-sm">
            <Lightning size={18} weight="fill" className="text-white" />
          </div>
          <div>
            <h1
              data-testid="app-title"
              className="text-lg font-bold text-white tracking-tight"
              style={{ fontFamily: "Outfit, sans-serif" }}
            >
              NER Compare
            </h1>
            <p className="text-[9px] font-mono text-gray-500 uppercase tracking-[0.2em]">
              spaCy vs GLiNER
            </p>
          </div>
        </div>
      </motion.div>

      {/* Center prompt when no country selected */}
      <AnimatePresence>
        {!selectedCountry && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: 0.8 }}
            className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10 text-center"
          >
            <div className="backdrop-blur-xl bg-black/50 border border-white/10 rounded-xl px-8 py-5">
              <div className="flex items-center justify-center gap-2.5 mb-3">
                <GlobeIcon size={20} weight="duotone" className="text-gray-400" />
                <p
                  className="text-sm font-medium text-gray-300"
                  style={{ fontFamily: "Outfit, sans-serif" }}
                >
                  Select a country to begin NER analysis
                </p>
              </div>
              <p className="text-[10px] font-mono text-gray-600 uppercase tracking-[0.15em] mb-3">
                Click on the globe or choose below
              </p>
              {countries.length > 0 && (
                <div className="relative">
                  <select
                    data-testid="country-dropdown"
                    onChange={(e) => {
                      if (e.target.value) handleCountrySelect(e.target.value);
                    }}
                    defaultValue=""
                    className="w-full appearance-none bg-white/5 border border-white/15 text-gray-300 text-sm rounded-lg px-4 py-2.5 pr-10 focus:outline-none focus:border-white/30 cursor-pointer font-sans"
                    style={{ fontFamily: "IBM Plex Sans, sans-serif" }}
                  >
                    <option value="" disabled className="bg-[#121212] text-gray-500">Quick select a country...</option>
                    {countries.map((c) => (
                      <option key={c} value={c} className="bg-[#121212] text-gray-300">{c}</option>
                    ))}
                  </select>
                  <CaretDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Country Panel (right side) */}
      <AnimatePresence>
        {selectedCountry && !showResults && (
          <CountryPanel
            country={selectedCountry}
            countryText={countryText}
            onClose={handleClosePanel}
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
          />
        )}
      </AnimatePresence>

      {/* Results Panel (full overlay) */}
      <AnimatePresence>
        {showResults && results && (
          <motion.div
            data-testid="results-overlay"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 z-30 backdrop-blur-2xl bg-black/80 border border-white/5"
          >
            <NERResults results={results} onClose={handleCloseResults} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;


