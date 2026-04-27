import React, { useRef, useState, useEffect, useCallback, useMemo } from "react";
import Globe from "react-globe.gl";

const GEOJSON_URL =
    "https://raw.githubusercontent.com/vasturiano/react-globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson";

export default function GlobeView({ onCountrySelect, selectedCountry }) {
    const globeRef = useRef();
    const [countries, setCountries] = useState({ features: [] });
    const [hovered, setHovered] = useState(null);
    const [dimensions, setDimensions] = useState({
        width: window.innerWidth,
        height: window.innerHeight,
    });

    useEffect(() => {
        fetch(GEOJSON_URL)
            .then((r) => r.json())
            .then((data) => setCountries(data));
    }, []);

    useEffect(() => {
        const handleResize = () => {
            setDimensions({ width: window.innerWidth, height: window.innerHeight });
        };
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    useEffect(() => {
        if (globeRef.current) {
            const controls = globeRef.current.controls();
            if (controls) {
                controls.autoRotate = !selectedCountry;
                controls.autoRotateSpeed = 0.4;
                controls.enableZoom = true;
                controls.minDistance = 200;
                controls.maxDistance = 600;
            }
            if (!selectedCountry) {
                globeRef.current.pointOfView({ lat: 20, lng: 10, altitude: 2.5 }, 1500);
            }
        }
    }, [selectedCountry]);

    useEffect(() => {
        if (globeRef.current) {
            const controls = globeRef.current.controls();
            if (controls) {
                controls.autoRotate = true;
                controls.autoRotateSpeed = 0.4;
            }
        }
    }, [countries]);

    const handleCountryClick = useCallback(
        (polygon) => {
            if (!polygon) return;
            const name = polygon.properties?.NAME || polygon.properties?.ADMIN;
            if (!name) return;

            const coords = polygon.properties;
            const lat = coords?.LABEL_Y || 0;
            const lng = coords?.LABEL_X || 0;

            if (globeRef.current) {
                globeRef.current.pointOfView({ lat, lng, altitude: 1.6 }, 1200);
            }

            onCountrySelect(name);
        },
        [onCountrySelect]
    );

    const handleHover = useCallback((polygon) => {
        setHovered(polygon);
    }, []);

    const capColor = useCallback(
        (d) => {
            const name = d.properties?.NAME || d.properties?.ADMIN;
            if (selectedCountry && name === selectedCountry)
                return "rgba(255, 255, 255, 0.25)";
            if (hovered && (hovered.properties?.NAME === name || hovered.properties?.ADMIN === name))
                return "rgba(255, 255, 255, 0.15)";
            return "rgba(255, 255, 255, 0.03)";
        },
        [hovered, selectedCountry]
    );

    const sideColor = useCallback(() => "rgba(255, 255, 255, 0.05)", []);
    const strokeColor = useCallback(() => "rgba(255, 255, 255, 0.12)", []);

    const polygonLabel = useCallback(
        (d) => {
            const name = d.properties?.NAME || d.properties?.ADMIN || "";
            return `<div style="font-family:Outfit,sans-serif;font-size:13px;font-weight:600;color:#fff;background:rgba(0,0,0,0.75);padding:6px 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.15);backdrop-filter:blur(8px)">${name}</div>`;
        },
        []
    );

    const globeImageUrl = useMemo(
        () => "//unpkg.com/three-globe/example/img/earth-dark.jpg",
        []
    );
    const bumpImageUrl = useMemo(
        () => "//unpkg.com/three-globe/example/img/earth-topology.png",
        []
    );

    return (
        <div data-testid="globe-container" className="absolute inset-0 z-0">
            <Globe
                ref={globeRef}
                width={dimensions.width}
                height={dimensions.height}
                globeImageUrl={globeImageUrl}
                bumpImageUrl={bumpImageUrl}
                backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
                polygonsData={countries.features}
                polygonCapColor={capColor}
                polygonSideColor={sideColor}
                polygonStrokeColor={strokeColor}
                polygonAltitude={(d) => {
                    const name = d.properties?.NAME || d.properties?.ADMIN;
                    return selectedCountry && name === selectedCountry ? 0.04 : 0.01;
                }}
                onPolygonClick={handleCountryClick}
                onPolygonHover={handleHover}
                polygonLabel={polygonLabel}
                atmosphereColor="rgba(100, 140, 255, 0.2)"
                atmosphereAltitude={0.25}
                animateIn={true}
            />
        </div>
    );
}


