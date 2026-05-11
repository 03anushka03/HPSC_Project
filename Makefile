# ============================================================
#  Makefile – 2D Ising Model OpenMP Monte Carlo Simulation
#  B23310 | High-Performance Scientific Computing
# ============================================================
#
#  Targets:
#    make all        – compile serial and parallel binaries
#    make serial     – compile serial binary only
#    make parallel   – compile parallel binary only
#    make run        – run both simulations
#    make plots      – generate all plots (requires Python 3 + matplotlib)
#    make paper      – compile IEEE LaTeX white paper to PDF
#    make clean      – remove compiled binaries and intermediate files
#    make distclean  – remove all generated files (binaries + data + plots)
# ============================================================

CXX       := g++
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra
OMP_FLAGS := -fopenmp

# Python interpreter
PYTHON    := python3

# LaTeX compiler
LATEX     := pdflatex
LATEXFLAGS := -interaction=nonstopmode

# ── Targets ─────────────────────────────────────────────────

.PHONY: all serial parallel run run_serial run_parallel plots paper clean distclean

all: serial parallel

serial: ising_serial
parallel: ising_parallel

ising_serial: ising_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Built: ising_serial"

ising_parallel: ising_parallel.cpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $< -o $@
	@echo "Built: ising_parallel"

# ── Run simulations ─────────────────────────────────────────

run: run_serial run_parallel

run_serial: ising_serial
	@echo "Running serial simulation..."
	./ising_serial

run_parallel: ising_parallel
	@echo "Running parallel simulation..."
	./ising_parallel

# ── Plots ───────────────────────────────────────────────────

plots: results_serial.txt plot_results.py make_paper_figures.py
	@echo "Generating plots..."
	$(PYTHON) make_paper_figures.py
	$(PYTHON) plot_results.py
	@echo "All plots written."

# ── White paper ─────────────────────────────────────────────

paper: ising_whitepaper.tex fig_magnetization.pdf fig_energy.pdf \
       fig_cv_chi.pdf fig_snapshots.pdf fig_speedup.pdf
	$(LATEX) $(LATEXFLAGS) ising_whitepaper.tex
	$(LATEX) $(LATEXFLAGS) ising_whitepaper.tex
	@echo "PDF written: ising_whitepaper.pdf"

# Generate figures if they don't exist yet
fig_magnetization.pdf fig_energy.pdf fig_cv_chi.pdf fig_snapshots.pdf \
fig_speedup.pdf: results_serial.txt make_paper_figures.py
	$(PYTHON) make_paper_figures.py

# ── Clean ───────────────────────────────────────────────────

clean:
	rm -f ising_serial ising_parallel
	rm -f ising_whitepaper.aux ising_whitepaper.log \
	      ising_whitepaper.out ising_whitepaper.bbl \
	      ising_whitepaper.blg ising_whitepaper.toc

distclean: clean
	rm -f results_serial.txt results_parallel_strategy1.txt \
	      results_parallel_strategy2.txt runtime_serial.txt \
	      runtime_parallel_s1.txt runtime_parallel_s2.txt \
	      speedup_data.txt snapshot_T*.txt
	rm -f *.png *.pdf
