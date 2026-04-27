import React from "react";
import NERTag from "./NERTag";
import { ScrollArea } from "../components/ui/scroll-area";

function AnnotatedText({ text, entities }) {
    if (!entities || entities.length === 0) {
        return <p className="text-gray-300 text-sm leading-relaxed font-sans">{text}</p>;
    }

    const sorted = [...entities].sort((a, b) => a.start - b.start);
    const fragments = [];
    let lastEnd = 0;

    sorted.forEach((entity, i) => {
        if (entity.start > lastEnd) {
            fragments.push(
                <span key={`text-${i}`} className="text-gray-300">
                    {text.slice(lastEnd, entity.start)}
                </span>
            );
        }
        fragments.push(<NERTag key={`entity-${i}`} entity={entity} />);
        lastEnd = entity.end;
    });

    if (lastEnd < text.length) {
        fragments.push(
            <span key="text-end" className="text-gray-300">
                {text.slice(lastEnd)}
            </span>
        );
    }

    return (
        <div className="text-sm leading-loose font-sans whitespace-pre-wrap">
            {fragments}
        </div>
    );
}

function EntitySummary({ entities }) {
    if (!entities || entities.length === 0) return null;
    const counts = {};
    entities.forEach((e) => {
        counts[e.label] = (counts[e.label] || 0) + 1;
    });

    return (
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-white/10">
            {Object.entries(counts).map(([label, count]) => (
                <span
                    key={label}
                    className="text-[10px] font-mono uppercase tracking-wider text-gray-500 bg-white/5 px-2 py-1 rounded"
                >
                    {label}: {count}
                </span>
            ))}
        </div>
    );
}

export default function NERResults({ results, onClose }) {
    if (!results) return null;

    const showBoth = results.model_type === "compare";
    const showSpacy = results.model_type === "spacy" || showBoth;
    const showGliner = results.model_type === "gliner" || showBoth;

    return (
        <div
            data-testid="ner-results-container"
            className="h-full flex flex-col"
        >
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
                <div>
                    <h2
                        className="text-xl font-semibold tracking-tight text-white"
                        style={{ fontFamily: "Outfit, sans-serif" }}
                    >
                        Inference Results
                    </h2>
                    <p className="text-xs font-mono text-gray-500 uppercase tracking-widest mt-1">
                        {results.country ? `${results.country} / ` : ""}
                        {showBoth ? "spaCy vs GLiNER" : showSpacy ? "spaCy Model" : "GLiNER Model"}
                    </p>
                </div>
                <button
                    data-testid="close-results-btn"
                    onClick={onClose}
                    className="text-gray-500 hover:text-white transition-colors p-2"
                >
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                </button>
            </div>

            <div
                className={`flex-1 overflow-hidden ${showBoth ? "grid grid-cols-1 md:grid-cols-2" : "grid grid-cols-1"
                    }`}
            >
                {showSpacy && (
                    <div
                        data-testid="spacy-results-panel"
                        className={`flex flex-col ${showBoth ? "border-r border-white/10" : ""}`}
                    >
                        <div className="px-6 py-3 border-b border-white/10 bg-sky-500/5">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-sky-400" />
                                <span
                                    className="text-sm font-semibold text-white tracking-tight"
                                    style={{ fontFamily: "Outfit, sans-serif" }}
                                >
                                    spaCy Model
                                </span>
                                <span className="text-[10px] font-mono text-gray-500 ml-auto">
                                    {results.spacy_entities?.length || 0} entities
                                </span>
                            </div>
                        </div>
                        <ScrollArea className="flex-1 p-6">
                            <AnnotatedText
                                text={results.text}
                                entities={results.spacy_entities}
                            />
                            <EntitySummary entities={results.spacy_entities} />
                        </ScrollArea>
                    </div>
                )}

                {showGliner && (
                    <div data-testid="gliner-results-panel" className="flex flex-col">
                        <div className="px-6 py-3 border-b border-white/10 bg-emerald-500/5">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-emerald-400" />
                                <span
                                    className="text-sm font-semibold text-white tracking-tight"
                                    style={{ fontFamily: "Outfit, sans-serif" }}
                                >
                                    GLiNER Model
                                </span>
                                <span className="text-[10px] font-mono text-gray-500 ml-auto">
                                    {results.gliner_entities?.length || 0} entities
                                </span>
                            </div>
                        </div>
                        <ScrollArea className="flex-1 p-6">
                            <AnnotatedText
                                text={results.text}
                                entities={results.gliner_entities}
                            />
                            <EntitySummary entities={results.gliner_entities} />
                        </ScrollArea>
                    </div>
                )}
            </div>
        </div>
    );
}


