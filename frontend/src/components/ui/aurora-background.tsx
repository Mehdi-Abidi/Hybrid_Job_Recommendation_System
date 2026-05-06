"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface AuroraBackgroundProps extends React.HTMLProps<HTMLDivElement> {
  children: ReactNode;
  showRadialGradient?: boolean;
}

export const AuroraBackground = ({
  className,
  children,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) => {
  return (
    <main>
      <div
        className={cn(
          "relative flex flex-col  h-[100vh] items-center justify-center bg-zinc-950 dark:bg-zinc-900  text-slate-950 transition-bg",
          className
        )}
        {...props}
      >
        <div className="absolute inset-0 overflow-hidden">
          <div
            className={cn(
              `
            [--white-gradient:repeating-linear-gradient(100deg,var(--white)_0%,var(--white)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--white)_16%)]
            [--dark-gradient:repeating-linear-gradient(100deg,var(--black)_0%,var(--black)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--black)_16%)]
            [--aurora:repeating-linear-gradient(100deg,#3b82f6_10%,#8b5cf6_20%,#06b6d4_30%,#10b981_40%,#3b82f6_50%)]
            [background-image:var(--dark-gradient),var(--aurora)]
            [background-size:300%,_200%]
            [background-position:50%_50%,50%_50%]
            filter blur-[10px] invert-0
            after:content-[""] after:absolute after:inset-0 after:[background-image:var(--dark-gradient),var(--aurora)] 
            after:[background-size:200%,_100%] 
            after:animate-aurora after:[background-attachment:fixed] after:mix-blend-difference
            pointer-events-none
            absolute -inset-[10px] opacity-20 will-change-transform`,
              showRadialGradient &&
                `[mask-image:radial-gradient(ellipse_at_100%_0%,black_10%,var(--transparent)_70%)]`
            )}
          ></div>
          {/* Fallback simple aurora for Tailwind v4 compatibility without complex plugins */}
          <div className="absolute -top-[50%] -left-[10%] w-[70vw] h-[70vw] rounded-full bg-teal-500/10 blur-[120px] mix-blend-screen pointer-events-none" />
          <div className="absolute top-[20%] -right-[10%] w-[50vw] h-[50vw] rounded-full bg-indigo-500/10 blur-[120px] mix-blend-screen pointer-events-none" />
          <div className="absolute -bottom-[20%] left-[20%] w-[60vw] h-[60vw] rounded-full bg-purple-500/10 blur-[120px] mix-blend-screen pointer-events-none" />
        </div>
        <div className="relative z-10 w-full h-full">{children}</div>
      </div>
    </main>
  );
};
