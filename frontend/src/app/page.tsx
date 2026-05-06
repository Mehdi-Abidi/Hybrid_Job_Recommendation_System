"use client";

import React, { useState } from "react";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { GlareCard } from "@/components/ui/glare-card";
import { FloatingDock } from "@/components/ui/floating-dock";
import { 
  Briefcase, 
  User, 
  GitBranch, 
  Sparkles, 
  Search,
  CheckCircle2,
  XCircle,
  ArrowRight
} from "lucide-react";
import { motion } from "framer-motion";

export default function Home() {
  const [activeJob, setActiveJob] = useState(0);

  const jobs = [
    {
      id: 1,
      title: "Senior ML Engineer",
      company: "Anthropic",
      location: "San Francisco, CA (Remote)",
      salary: "$190k - $240k",
      match: "98%",
      matchType: "Perfect Match",
      reason: "Your deep expertise in PyTorch directly aligns with Anthropic's current LLM scaling requirements. Your background in distributed training systems perfectly fills their immediate skill gap.",
      skills: [
        { name: "PyTorch", status: "match" },
        { name: "Distributed Systems", status: "match" },
        { name: "Kubernetes", status: "adjacent", via: "Docker" }
      ]
    },
    {
      id: 2,
      title: "Lead AI Architect",
      company: "NeuroFlow",
      location: "Remote",
      salary: "$175k - $210k",
      match: "94%",
      matchType: "High Potential",
      reason: "NeuroFlow is looking for leadership in generative models. Your recent transition from Senior ML Engineer suggests readiness for an Architect role.",
      skills: [
        { name: "NLP", status: "match" },
        { name: "System Architecture", status: "match" },
        { name: "Team Lead", status: "missing" }
      ]
    }
  ];

  const dockItems = [
    { title: "Job Feed", icon: <Briefcase className="size-5" />, href: "#" },
    { title: "Profile", icon: <User className="size-5" />, href: "#" },
    { title: "Pipeline Inspector", icon: <GitBranch className="size-5" />, href: "#" },
    { title: "Search (Cmd+K)", icon: <Search className="size-5" />, href: "#" },
  ];

  return (
    <AuroraBackground>
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pt-20 pb-32">
        
        {/* Header / Omnibar Area */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-12 w-full max-w-2xl px-6"
        >
          <div className="flex items-center gap-4 bg-white/[0.05] border border-white/10 backdrop-blur-2xl rounded-full p-2 pl-6 shadow-2xl">
            <Sparkles className="size-5 text-indigo-400" />
            <input 
              type="text" 
              placeholder="Ask the AI for specific roles... (Cmd+K)" 
              className="bg-transparent border-none text-white w-full focus:outline-none text-sm font-medium placeholder:text-white/40"
            />
            <button className="bg-white text-black px-4 py-1.5 rounded-full text-xs font-bold tracking-wide">
              Search
            </button>
          </div>
        </motion.div>

        {/* Spatial Carousel */}
        <div className="w-full max-w-[1200px] flex items-center justify-center gap-12 mt-10">
          
          {/* Main Focused Card */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-[450px] h-[600px] z-20 perspective-1000"
          >
            <GlareCard className="w-full h-full flex flex-col p-8 group">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <h2 className="text-3xl font-bold tracking-tight text-white mb-2 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-white/60 transition-colors">
                    {jobs[activeJob].title}
                  </h2>
                  <p className="text-lg text-indigo-300 font-medium">{jobs[activeJob].company}</p>
                </div>
                <div className="flex flex-col items-end">
                  <div className="text-3xl font-black text-white">{jobs[activeJob].match}</div>
                  <div className="text-xs font-bold uppercase tracking-widest text-teal-400 mt-1">{jobs[activeJob].matchType}</div>
                </div>
              </div>

              <div className="space-y-4 mb-8">
                <div className="flex items-center justify-between text-white/70 border-b border-white/10 pb-4">
                  <span className="text-sm font-medium">Location</span>
                  <span className="text-sm text-white">{jobs[activeJob].location}</span>
                </div>
                <div className="flex items-center justify-between text-white/70 border-b border-white/10 pb-4">
                  <span className="text-sm font-medium">Predicted Salary</span>
                  <span className="text-sm text-white font-medium">{jobs[activeJob].salary}</span>
                </div>
              </div>

              <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 mb-8 backdrop-blur-sm shadow-inner relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500/50" />
                <h3 className="text-xs font-bold text-indigo-300 uppercase tracking-wider mb-2">Bilateral Match Reasoning</h3>
                <p className="text-sm text-white/80 leading-relaxed font-medium">
                  {jobs[activeJob].reason}
                </p>
              </div>

              <div className="mt-auto">
                <button className="w-full py-4 rounded-xl bg-white text-black font-bold text-sm hover:bg-neutral-200 transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_rgba(255,255,255,0.5)]">
                  Apply with 1-Click
                  <ArrowRight className="size-4" />
                </button>
              </div>
            </GlareCard>
          </motion.div>

          {/* Skill Constellation Graph (Abstracted visually for the panel) */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="w-[350px] h-[500px] bg-white/[0.02] border border-white/10 rounded-[32px] p-6 backdrop-blur-2xl flex flex-col relative overflow-hidden"
          >
            {/* Ambient graph glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-indigo-500/10 blur-[80px] rounded-full pointer-events-none" />
            
            <h3 className="text-lg font-bold text-white mb-6">Skill Constellation</h3>
            
            <div className="flex-1 flex flex-col justify-center gap-6 relative z-10">
              {jobs[activeJob].skills.map((skill, i) => (
                <div key={i} className="flex items-center gap-4 group cursor-default">
                  <div className="relative">
                    {skill.status === "match" ? (
                      <CheckCircle2 className="size-6 text-teal-400 drop-shadow-[0_0_10px_rgba(45,212,191,0.5)]" />
                    ) : skill.status === "missing" ? (
                      <XCircle className="size-6 text-red-400 drop-shadow-[0_0_10px_rgba(248,113,113,0.5)]" />
                    ) : (
                      <div className="size-6 rounded-full border-2 border-indigo-400 border-dashed flex items-center justify-center">
                        <div className="size-2 rounded-full bg-indigo-400" />
                      </div>
                    )}
                    {/* Fake Beam effect */}
                    {i !== jobs[activeJob].skills.length - 1 && (
                      <div className="absolute top-6 left-1/2 -translate-x-1/2 w-[2px] h-6 bg-gradient-to-b from-white/20 to-transparent" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-white">{skill.name}</p>
                    <p className="text-xs text-white/50 capitalize">
                      {skill.status} {skill.via && `(via ${skill.via})`}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-white/10 relative z-10">
              <p className="text-xs text-white/60 leading-relaxed text-center font-medium">
                Your ontology nodes align perfectly. You are in the top 1% of applicants for this stack.
              </p>
            </div>
          </motion.div>

        </div>
      </div>

      <FloatingDock items={dockItems} />
    </AuroraBackground>
  );
}
