  Reusable from main.tex (high → low value)                                                                                                
                                                                                                                                           
  Section 1 (Methods/Background) — directly portable                                                                                       
                                                                                                                                           
  Section 1.1 Steering LLM Behavior (line 100-104): clean CAA description with citations (turner2023activation, rimsky2023steering,        
  tan2024analyzing). Reuse verbatim as your CAA paragraph.
                                                                                                                                           
  Section 1.2 Unsupervised Steering (lines 106-114): MELBO description with the equation and motivation. Reuse verbatim as MELBO paragraph.
   Cite mack2024mechanistically. Also has the natural transition into "what does the linear approximation tell us" that introduces PI.
                                                                                                                                           
  Section 1.3 Power Steering (lines 123-192): notation, Jacobian definition, the four-step JtJ recipe, math footnote on convergence,       
  Algorithm 1 (block power iteration with Rayleigh-Ritz), the PyTorch code listing. All reusable as Section 2. The notation block at line
  125-129 is especially well-formed.                                                                                                       
                                          
  The figure powersteer-scheme.png (the CAA/MELBO/PS comparison schematic) is valuable — same diagram applies to the new paper.            
   
  Section 2 (Mapping an Entire Model) — high reuse for new Section 4                                                                       
                                          
  Lines 194-211 are essentially the atlas-section setup. The KL-heatmap discussion, scale-norm tuning rationale (0.35 × residual norm), and
   sign-sensitivity caveat all transfer directly. Update Qwen3-8B → Qwen3-14B and update prompt count.
                                                                                                                                           
  Section 3.1 Anti-Refusal Steering (lines 256-274) — direct port to new Section 4                                                         
   
  Anti-refusal prompt setup, "top singular vector across many source layers produces anti-refusal," cosine-similarity-within-source-layer  
  observation ((14,27)v0 and (14,19)v0 cosine 0.96), the two example generation footnotes (educational hedge vs no hedge). All directly 
  portable for the new Section 4.                                                                                                          
                                          
  Algorithm + code listing (lines 143-190) — keep as-is                                                                                    
   
  The block-power-iteration algorithm and PyTorch reference are both ~half-page each. Both can move to appendix if main body is tight, or  
  stay in Section 2 if there's space.     
                                                                                                                                           
  Limitations (lines 410-424) — partially reusable                                                                                         
   
  - "Sign sensitivity" — keep                                                                                                              
  - "Scale selection underexplored" — partly addressed now (we have the moderate-scale comparison) but still worth mentioning
  - "KL is noisy proxy" — REPLACE with "first-token logit metrics overstate compliance / regex undercounts under generation" — that's the  
  new methodological note                                                                                                                  
  - "Source/target not optimized" — REPLACE since we now have data showing it matters                                                      
  - "Method finds directions network is sensitive to" — keep as conceptual limitation                                                      
  - "Dataset contamination" — keep                                                                                                         
                                                                                                                                           
  Discard or move to appendix                                                                                                              
                                                                                                                                           
  - Section 3.2 Corrigibility Steering in Qwen3-14B (lines 276-326): discard the analysis but keep the figures' style as a template. The   
  data is now superseded by your 7-eval comparison. The violin plot figures violin_selected.png, generation_by_dataset_selected.png,
  unclear_by_dataset_selected.png, melbo_vs_power_steering.png are obsolete.                                                               
  - Section 3.3 CoT Elicitation in Qwen3-1.7B-Base (lines 328-390): move to appendix as a separate "Latent behavior on small models"
  subsection. It supports the broader "PS finds latent behavior" claim but isn't load-bearing for the new paper's headline.                
  - Section 3.4 Inducing Refusal in Roleplay (lines 392-408): move to appendix. Same rationale.
                                                                                                                                           
  Concrete porting plan                   
                                                                                                                                           
  1. Keep: Section 1.1, 1.2, 1.3 (most of it), the algorithm + listing, the schematic figure, anti-refusal qualitative content,            
  sign-sensitivity & scale notes.
  2. Replace: Abstract (have new draft), Section 3.2 corrigibility data (use new 7-eval data), most of Limitations (write fresh ones).     
  3. Add new: Cross-eval comparison Section 3 (3 methods, logit + LLM-judged), Section 4 atlas + AdvBench logit-vs-content finding, new    
  figures (per-method heatmap grids, AdvBench bar chart with compliance diamonds).                                                         
  4. Move to appendix: CoT, refusal-on-roleplay, DCT comparison, full Chinese-vector details. 