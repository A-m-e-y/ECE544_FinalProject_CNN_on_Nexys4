import './App.scss';
import 'primeicons/primeicons.css';
import 'primereact/resources/primereact.min.css';
import backgroundImage from './images/cnn_background_image.png';
import { Button } from 'primereact/button';

import React, { useRef, useState, useEffect } from "react";

const App = () => {
    const canvasARef = useRef(null);
    const [result, setResult] = useState([]);
    const [showPlaceholder, setShowPlaceholder] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [fpgaDigit, setFpgaDigit] = useState(null); 

    const useDrawing = (canvasRef) => {
        useEffect(() => {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = 15;
            ctx.strokeStyle = "maroon";
    
            let drawing = false;
    
            const getPos = (e) => {
                if (e.touches && e.touches.length > 0) {
                    const rect = canvas.getBoundingClientRect();
                    console.log(rect);
                    return {
                        x: e.touches[0].clientX - rect.left,
                        y: e.touches[0].clientY - rect.top,
                    };
                }
                return {
                    x: e.offsetX,
                    y: e.offsetY,
                };
            };
    
            const start = (e) => {
                setShowPlaceholder(false);
                drawing = true;
                const pos = getPos(e);
                ctx.beginPath();
                ctx.moveTo(pos.x, pos.y);
            };
    
            const move = (e) => {
                if (!drawing) return;
                const pos = getPos(e);
                
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
            };
    
            const end = () => {
                drawing = false;
                ctx.closePath();
            };
    
            canvas.addEventListener("mousedown", start);
            canvas.addEventListener("mousemove", move);
            window.addEventListener("mouseup", end);
        
            canvas.addEventListener("touchstart", start, { passive: false });
            canvas.addEventListener("touchmove", (e) => {
                e.preventDefault();
                move(e);
            }, { passive: false });
            canvas.addEventListener("touchend", end);
    
            return () => {
                canvas.removeEventListener("mousedown", start);
                canvas.removeEventListener("mousemove", move);
                window.removeEventListener("mouseup", end);
        
                canvas.removeEventListener("touchstart", start);
                canvas.removeEventListener("touchmove", move);
                canvas.removeEventListener("touchend", end);
            };
        }, [canvasRef]);
    };

    useEffect(() => {
        if (!canvasARef.current) return;

        const canvas = canvasARef.current;
        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (showPlaceholder) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            ctx.font = "20px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Draw a digit (0–9)", canvas.width / 2, canvas.height / 2);
        }
        }, [showPlaceholder]);
    
    useDrawing(canvasARef);
    
    const processCanvas = (canvasRef) => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const img = ctx.getImageData(0, 0, 200, 200);
        const data = img.data;
        setFpgaDigit(null);
        
        // Convert to grayscale
        const gray = [];
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const y = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            gray.push(y);
        }
        
        // Downsample 200x200 → 10x10 (block averaging 20x20)
        const block = 20;
        const out = [];
        for (let y = 0; y < 10; y++) {
            for (let x = 0; x < 10; x++) {
                let sum = 0;
                for (let yy = 0; yy < block; yy++) {
                    for (let xx = 0; xx < block; xx++) {
                        const px = (y * block + yy) * 200 + (x * block + xx);
                        sum += gray[px];
                    }
                }
                out.push(Math.round(sum / (block * block)));
            }
        }
        
        return out;
    };
    
    const handleProcess = () => {
        setResult(processCanvas(canvasARef));
    };
    
    const clearCanvas = (ref) => {
        const ctx = ref.current.getContext("2d");
        ctx.clearRect(0, 0, 200, 200);
        setShowPlaceholder(true);
        setFpgaDigit(null);
        setResult([]);
    };

    const backgroundStyle = {
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        minHeight: '100vh',
    };

    const sendToFpgaUART = async (array10x10) => {
        return new Promise((resolve) => {
            setTimeout(() => {
                const simulatedDigit = Math.floor(Math.random() * 10);
                resolve(simulatedDigit);
            }, 3000); 
        });
    };

    const handleSubmit = async () => {
        setIsProcessing(true);
        setFpgaDigit(null);

        const digit = await sendToFpgaUART(result);

        setFpgaDigit(digit);
        setIsProcessing(false);
        setResult([]);
    };
    
    return (
    <div className="App min-h-screen bg-gray-900 bg-blend-overlay" style={backgroundStyle}>
        <div className="max-w-fit mx-auto pt-20">

            <p className="bg-gradient-to-r from-fuchsia-400 via-purple-500 to-indigo-400 
                        bg-clip-text text-transparent text-7xl font-extrabold 
                        drop-shadow-[0_0_25px_rgba(209,0,255,0.4)]
                        pb-14 text-center">
                Welcome to ECE 544 Project Demo
            </p>

            <div className="flex justify-center mb-10">
                <div className="p-4 bg-gray-900/40 rounded-2xl shadow-[0_0_25px_rgba(139,92,246,0.4)] 
                                border border-purple-500/40 backdrop-blur-xl">
                    <canvas
                        ref={canvasARef}
                        width={201}
                        height={201}
                        className="rounded-xl shadow-[0_0_35px_rgba(255,255,255,0.2)] bg-white touch-none"
                    />
                </div>
            </div>

            <div className="flex gap-6 justify-center mb-12">
                <Button
                    label="Clear"
                    onClick={() => clearCanvas(canvasARef)}
                    disabled={isProcessing || showPlaceholder}
                    raised
                    className="
                    !bg-gradient-to-r !from-amber-400 !to-yellow-500 
                    !text-gray-900 !font-semibold 
                    !shadow-[0_0_15px_rgba(255,200,0,0.6)]
                    !rounded-xl !h-12 !w-40 !text-lg
                    hover:!shadow-[0_0_25px_rgba(255,220,50,0.8)]
                    transition-all"
                />

                <Button
                    label="Process Image"
                    onClick={handleProcess}
                    disabled={isProcessing || showPlaceholder}
                    raised
                    className="
                    !bg-gradient-to-r !from-indigo-500 !to-purple-600 
                    !text-white !font-semibold 
                    !shadow-[0_0_20px_rgba(147,51,234,0.7)]
                    !rounded-xl !h-12 !w-48 !text-lg
                    hover:!shadow-[0_0_30px_rgba(168,85,247,0.9)]
                    transition-all"
                />

                <Button
                    label="Submit"
                    onClick={handleSubmit}
                    disabled={isProcessing || result.length === 0}
                    raised
                    className="
                    !bg-gradient-to-r !from-green-400 !to-emerald-600
                    !text-white !font-semibold
                    !shadow-[0_0_20px_rgba(16,185,129,0.7)]
                    !rounded-xl !h-12 !w-40 !text-lg
                    hover:!shadow-[0_0_30px_rgba(52,211,153,0.9)]
                    transition-all"
                />
            </div>

            <div className="flex gap-10 justify-center items-start mt-10">

                { (result.length !== 0 || fpgaDigit) && <div className="
                    p-4 bg-gray-900/30 rounded-xl 
                    shadow-[0_0_25px_rgba(100,100,255,0.3)]
                    border border-indigo-500/40
                    backdrop-blur-xl
                    w-fit
                ">

                    {isProcessing ? (
                        <div className="flex flex-col items-center justify-center px-20 py-10">
                            <i className="pi pi-spin pi-spinner text-6xl text-indigo-400"></i>
                            <p className="text-white mt-4 text-lg">Processing...</p>
                        </div>

                    ) : fpgaDigit !== null ? (
                        <div className="flex flex-col items-center">
                            <p
                                className="text-white font-bold mb-4 drop-shadow-lg"
                                style={{ fontSize: '10rem', lineHeight: '1' }}
                            >
                                {fpgaDigit}
                            </p>
                            <p className="text-white text-xl font-bold">
                                FPGA Output: {fpgaDigit}
                            </p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-10 gap-1">
                            {result.map((value, index) => (
                                <div
                                    key={index}
                                    className="w-10 h-10 flex items-center justify-center 
                                               bg-white/90 border border-gray-300 rounded-md 
                                               text-sm font-medium text-gray-900 shadow-sm">
                                    {value}
                                </div>
                            ))}
                        </div>
                    )}

                </div>}
            </div>

        </div>
    </div>
    );
}

export default App;
   