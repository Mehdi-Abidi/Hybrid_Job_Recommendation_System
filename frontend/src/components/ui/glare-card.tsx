"use client";

import { cn } from "@/lib/utils";
import { useRef, useState } from "react";

export const GlareCard = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) => {
  const isPointerInside = useRef(false);
  const refElement = useRef<HTMLDivElement>(null);
  const state = useRef({
    glare: { x: 50, y: 50 },
    background: { x: 50, y: 50 },
    rotate: { x: 0, y: 0 },
  });
  const [styles, setStyles] = useState({
    "--m-x": "50%",
    "--m-y": "50%",
    "--r-x": "0deg",
    "--r-y": "0deg",
    "--bg-x": "50%",
    "--bg-y": "50%",
    "--duration": "300ms",
    "--foil-size": "100%",
    "--opacity": "0",
    "--radius": "24px",
    "--step": "5%",
    "--pattern": "linear-gradient(135deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%)",
  } as React.CSSProperties);

  const containerStyle = {
    "--m-x": "50%",
    "--m-y": "50%",
    "--r-x": "0deg",
    "--r-y": "0deg",
    "--bg-x": "50%",
    "--bg-y": "50%",
    "--duration": "300ms",
    "--foil-size": "100%",
    "--opacity": "0",
    "--radius": "24px",
    "--step": "5%",
    "--pattern": "linear-gradient(135deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%)",
  } as React.CSSProperties;

  const updateStyles = () => {
    if (refElement.current) {
      const { background, rotate, glare } = state.current;
      setStyles((s) => ({
        ...s,
        "--m-x": `${glare.x}%`,
        "--m-y": `${glare.y}%`,
        "--r-x": `${rotate.x}deg`,
        "--r-y": `${rotate.y}deg`,
        "--bg-x": `${background.x}%`,
        "--bg-y": `${background.y}%`,
        "--opacity": isPointerInside.current ? "1" : "0",
        "--duration": isPointerInside.current ? "0ms" : "300ms",
      }));
    }
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const rotateFactor = 0.4;
    const rect = event.currentTarget.getBoundingClientRect();
    const position = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    const percentage = {
      x: (100 / rect.width) * position.x,
      y: (100 / rect.height) * position.y,
    };
    const delta = {
      x: percentage.x - 50,
      y: percentage.y - 50,
    };

    const { background, rotate, glare } = state.current;
    background.x = 50 + percentage.x / 4 - 12.5;
    background.y = 50 + percentage.y / 3 - 16.67;
    rotate.x = -(delta.y * rotateFactor);
    rotate.y = delta.x * rotateFactor;
    glare.x = percentage.x;
    glare.y = percentage.y;

    updateStyles();
  };

  const handlePointerEnter = () => {
    isPointerInside.current = true;
    setTimeout(() => {
      if (isPointerInside.current) {
        refElement.current?.style.setProperty("--duration", "0ms");
      }
    }, 300);
  };

  const handlePointerLeave = () => {
    isPointerInside.current = false;
    refElement.current?.style.setProperty("--duration", "300ms");
    state.current = {
      glare: { x: 50, y: 50 },
      background: { x: 50, y: 50 },
      rotate: { x: 0, y: 0 },
    };
    updateStyles();
  };

  return (
    <div
      style={styles}
      className={cn("relative isolate [contain:layout_style] [perspective:600px] transition-transform duration-[var(--duration)] ease-[ease-out] will-change-transform", className)}
      ref={refElement}
      onPointerMove={handlePointerMove}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
    >
      <div className="h-full grid will-change-transform origin-center transition-transform duration-[var(--duration)] ease-[ease-out] [transform:rotateY(var(--r-x))_rotateX(var(--r-y))] rounded-[var(--radius)] border border-white/10 hover:[--opacity:0.6] hover:[--duration:200ms] hover:[--ease:linear] hover:filter-none overflow-hidden">
        <div className="w-full h-full grid [grid-area:1/1] mix-blend-color-dodge opacity-[var(--opacity)] will-change-background transition-opacity [clip-path:inset(0_0_1px_0_round_var(--radius))] [background-blend-mode:hue_hue_hue_overlay] [background:var(--pattern),_var(--rainbow),_var(--diagonal),_var(--shade)] relative after:content-[''] after:grid-area-[inherit] after:bg-repeat-[inherit] after:bg-attachment-[inherit] after:bg-origin-[inherit] after:bg-clip-[inherit] after:bg-[inherit] after:mix-blend-exclusion after:[background-size:var(--foil-size),_200%_400%,_800%,_200%] after:[background-position:center,_0%_var(--bg-y),_calc(var(--bg-x)*_-1)_calc(var(--bg-y)*_-1),_var(--bg-x)_var(--bg-y)] after:[background-blend-mode:soft-light,_hue,_hard-light]" />
        
        {/* Soft Glare layer */}
        <div className="absolute inset-0 z-50 pointer-events-none rounded-[var(--radius)] opacity-[var(--opacity)] transition-opacity duration-300">
           <div className="absolute inset-0 bg-[radial-gradient(farthest-corner_circle_at_var(--m-x)_var(--m-y),rgba(255,255,255,0.1)_10%,rgba(255,255,255,0)_40%)] mix-blend-overlay" />
        </div>
        
        {/* Glassmorphic Panel */}
        <div className="w-full h-full relative z-10 [grid-area:1/1] bg-white/[0.02] backdrop-blur-md rounded-[var(--radius)] border-t border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]">
            {children}
        </div>
      </div>
    </div>
  );
};
