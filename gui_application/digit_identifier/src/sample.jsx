import React, { useRef, useState, useEffect } from "react";

// A cleaner Tailwind-styled dual drawing editor with touch + mouse support.
// Each canvas is 200x200. A button reads pixel data, converts to grayscale,
// downsamples to 10x10, and stores arrays.

export default function DualDrawingEditor() {
  const canvasARef = useRef(null);
  const canvasBRef = useRef(null);
  const [resultA, setResultA] = useState([]);
  const [resultB, setResultB] = useState([]);

  // ---- Helper: set up drawing handlers (mouse + touch) ----
  const useDrawing = (canvasRef) => {
    useEffect(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = 8;
      ctx.strokeStyle = "black";

      let drawing = false;

      const getPos = (e) => {
        if (e.touches && e.touches.length > 0) {
          const rect = canvas.getBoundingClientRect();
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

  useDrawing(canvasARef);
  useDrawing(canvasBRef);

  // ---- Convert to grayscale + downsample to 10x10 ----
  const processCanvas = (canvasRef) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = ctx.getImageData(0, 0, 200, 200);
    const data = img.data;

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
    setResultA(processCanvas(canvasARef));
    setResultB(processCanvas(canvasBRef));
  };

  const clearCanvas = (ref) => {
    const ctx = ref.current.getContext("2d");
    ctx.clearRect(0, 0, 200, 200);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-10 px-4">
      <h1 className="text-2xl font-semibold mb-6">Dual Drawing Editor</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Canvas A */}
        <div className="flex flex-col items-center gap-3">
          <canvas
            ref={canvasARef}
            width={200}
            height={200}
            className="border rounded shadow bg-white touch-none"
          />
          <button
            onClick={() => clearCanvas(canvasARef)}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm"
          >
            Clear A
          </button>
        </div>

        {/* Canvas B */}
        <div className="flex flex-col items-center gap-3">
          <canvas
            ref={canvasBRef}
            width={200}
            height={200}
            className="border rounded shadow bg-white touch-none"
          />
          <button
            onClick={() => clearCanvas(canvasBRef)}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm"
          >
            Clear B
          </button>
        </div>
      </div>

      <button
        onClick={handleProcess}
        className="px-6 py-2 bg-indigo-600 text-white rounded shadow mb-8 hover:bg-indigo-700"
      >
        Process Images
      </button>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-2xl">
        <div>
          <h2 className="font-medium mb-2">Canvas A 10x10 Data:</h2>
          <pre className="bg-white p-3 rounded border text-xs overflow-auto h-48">{JSON.stringify(resultA, null, 2)}</pre>
        </div>
        <div>
          <h2 className="font-medium mb-2">Canvas B 10x10 Data:</h2>
          <pre className="bg-white p-3 rounded border text-xs overflow-auto h-48">{JSON.stringify(resultB, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
