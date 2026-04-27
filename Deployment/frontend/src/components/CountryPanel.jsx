import React, { useState } from "react";
import { motion } from "framer-motion";
import { MagnifyingGlass, ArrowLeft, Brain, CircleNotch } from "@phosphor-icons/react";
import { NER_COLORS } from "../utils/nerColors";

export default function CountryPanel({
    country,
    countryText,
    onClose,
    onAnalyze,
    isLoading,
}) {
    const [text, setText] = useState(countryText || "");
    const [modelType, setModelType] = useState("compare");

    React.useEffect(() => {
        if (countryText) setText(countryText);
    }, [countryText]);

    const handleAnalyze = () => {
        if (!text.trim()) return;
        onAnalyze(text, modelType);
    };

    return (
        <motion.div
            data-testid="country-panel"
            initial={{ x: "100%", opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-full sm:w-[480px] z-20 backdrop-blur-2xl bg-black/70 border-l border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.5)] flex flex-col"
        >
            {/* Header */}
            <div className="flex items-center gap-3 px-6 py-5 border-b border-white/10">
                <button
                    data-testid="close-panel-btn"
                    onClick={onClose}
                    className="text-gray-400 hover:text-white transition-colors"
                >
                    <ArrowLeft size={20} weight="bold" />
                </button>
                <div>
                    <h2
                        data-testid="country-name"
                        className="text-xl font-semibold text-white tracking-tight"
                        style={{ fontFamily: "Outfit, sans-serif" }}
                    >
                        {country}
                    </h2>
                    <p className="text-[10px] font-mono text-gray-500 uppercase tracking-[0.15em] mt-0.5">
                        Political & Economic Context
                    </p>
                </div>
            </div>

            {/* Chat-like text area */}
            <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
                <div className="space-y-2">
                    <label className="text-[10px] font-mono text-gray-500 uppercase tracking-[0.15em]">
                        Generated Context
                    </label>
                    <div className="bg-white/5 border border-white/10 rounded-lg p-3 mb-2">
                        <p className="text-xs text-gray-500 font-mono mb-1">SYSTEM</p>
                        <p className="text-sm text-gray-400 leading-relaxed">
                            Auto-generated political and economic context for {country}. You can edit or replace the text below.
                        </p>
                    </div>
                    <textarea
                        data-testid="text-input"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Enter or modify text for NER analysis..."
                        className="w-full min-h-[200px] bg-white/5 border border-white/10 text-white rounded-lg p-4 font-sans text-sm focus:outline-none focus:border-white/30 focus:ring-1 focus:ring-white/30 transition-all resize-y placeholder:text-gray-600"
                    />
                    <p className="text-[10px] font-mono text-gray-600">
                        {text.length} characters
                    </p>
                </div>

                {/* Model selection */}
                <div className="space-y-3">
                    <label className="text-[10px] font-mono text-gray-500 uppercase tracking-[0.15em]">
                        Model Selection
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                        {[
                            { value: "spacy", label: "spaCy", color: "sky" },
                            { value: "gliner", label: "GLiNER", color: "emerald" },
                            { value: "compare", label: "Compare", color: "white" },
                        ].map((m) => (
                            <button
                                key={m.value}
                                data-testid={`model-btn-${m.value}`}
                                onClick={() => setModelType(m.value)}
                                className={`py-2.5 px-3 rounded-lg text-xs font-semibold font-mono uppercase tracking-wider transition-all border ${modelType === m.value
                                        ? m.color === "sky"
                                            ? "bg-sky-500/20 border-sky-500/50 text-sky-400"
                                            : m.color === "emerald"
                                                ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400"
                                                : "bg-white/10 border-white/30 text-white"
                                        : "bg-transparent border-white/10 text-gray-500 hover:text-gray-300 hover:border-white/20"
                                    }`}
                            >
                                {m.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* NER Labels Legend */}
                <div className="space-y-2">
                    <label className="text-[10px] font-mono text-gray-500 uppercase tracking-[0.15em]">
                        Entity Labels
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                        {Object.entries(NER_COLORS).map(([label, colors]) => (
                            <span
                                key={label}
                                className={`inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider border ${colors.bg} ${colors.border} ${colors.text}`}
                            >
                                {label}
                            </span>
                        ))}
                    </div>
                </div>
            </div>

            {/* Run button */}
            <div className="px-6 py-5 border-t border-white/10">
                <button
                    data-testid="run-inference-btn"
                    onClick={handleAnalyze}
                    disabled={isLoading || !text.trim()}
                    className="w-full bg-white text-black font-bold font-sans py-3.5 px-4 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center gap-2.5 disabled:opacity-40 disabled:cursor-not-allowed group"
                    style={{ fontFamily: "Outfit, sans-serif" }}
                >
                    {isLoading ? (
                        <>
                            <CircleNotch size={18} className="animate-spin" />
                            Running Inference...
                        </>
                    ) : (
                        <>
                            <Brain size={18} weight="bold" className="group-hover:scale-110 transition-transform" />
                            Run NER Inference
                            <MagnifyingGlass size={14} weight="bold" className="opacity-50" />
                        </>
                    )}
                </button>
            </div>
        </motion.div>
    );
}


