import React from "react";
import { NER_COLORS } from "../utils/nerColors";

export default function NERTag({ entity }) {
    const colors = NER_COLORS[entity.label] || NER_COLORS.GPE;

    return (
        <span
            data-testid={`ner-tag-${entity.label.toLowerCase()}`}
            className={`inline-flex items-center rounded px-1.5 py-0.5 mx-0.5 text-xs font-mono font-medium border leading-tight gap-1 ${colors.bg} ${colors.border} ${colors.text}`}
        >
            <span>{entity.text}</span>
            <span className="text-[9px] opacity-70 uppercase font-bold tracking-wider">
                {entity.label}
            </span>
        </span>
    );
}

