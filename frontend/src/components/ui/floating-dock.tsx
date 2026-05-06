"use client";

import { cn } from "@/lib/utils";
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useRef, useState } from "react";

export const FloatingDock = ({
  items,
  desktopClassName,
}: {
  items: { title: string; icon: React.ReactNode; href: string }[];
  desktopClassName?: string;
  mobileClassName?: string;
}) => {
  return (
    <div className={cn("fixed bottom-8 left-1/2 -translate-x-1/2 z-50", desktopClassName)}>
      <div className="mx-auto flex h-16 items-end gap-4 rounded-2xl bg-[#09090b]/80 backdrop-blur-xl border border-white/10 px-4 pb-3 shadow-[0_0_40px_rgba(0,0,0,0.5)]">
        {items.map((item, idx) => (
          <IconContainer key={item.title} {...item} />
        ))}
      </div>
    </div>
  );
};

function IconContainer({ title, icon, href }: { title: string; icon: React.ReactNode; href: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState(false);

  return (
    <div
      ref={ref}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative flex aspect-square h-10 w-10 items-center justify-center rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition-colors cursor-pointer group"
    >
      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 10, x: "-50%" }}
            animate={{ opacity: 1, y: 0, x: "-50%" }}
            exit={{ opacity: 0, y: 2, x: "-50%" }}
            className="absolute -top-10 left-1/2 w-fit whitespace-pre rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-white border border-white/10 shadow-xl"
          >
            {title}
          </motion.div>
        )}
      </AnimatePresence>
      <div className="text-white/70 group-hover:text-white transition-colors">
        {icon}
      </div>
    </div>
  );
}
