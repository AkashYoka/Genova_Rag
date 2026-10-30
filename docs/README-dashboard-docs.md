# Genova RAG — Dashboard Documentation

## Converting Markdown to PDF

This directory contains the complete documentation for the `/dashboard` page of the Genova RAG application.

### Files

| File | Description |
|---|---|
| `dashboard-functionality.md` | Complete markdown documentation (this file) |
| `dashboard-functionality.css` | Professional PDF-friendly stylesheet |
| `README-dashboard-docs.md` | This file — conversion instructions |

### Prerequisites

#### Option 1: Using Pandoc (recommended)

Pandoc is a universal document converter that supports Markdown, PDF output, custom CSS, and Mermaid diagrams.

**Installation:**
```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install pandoc

# Windows (via Chocolatey)
choco install pandoc
```

**Additional tools for Mermaid diagrams:**
```bash
# Install the mermaid-cli globally
npm install -g mermaid-cli
```
Or use the `mermaid-filter` pandoc filter:
```bash
# macOS
brew install mermaid-cli

# Or via pip
pip install mermaid-py
```

**Conversion command:**
```bash
pandoc docs/dashboard-functionality.md \
  -o docs/dashboard-functionality.pdf \
  --css=docs/dashboard-functionality.css \
  --pdf-engine=xelatex \
  --toc \
  --information-string="Author: Genova RAG Documentation" \
  --variable=geometry:margin=2cm
```

#### Option 2: Using WeasyPrint

```bash
pip install weasyprint
```

```bash
# Convert markdown to HTML first, then to PDF
pandoc docs/dashboard-functionality.md -o dashboard.html --css=docs/dashboard-functionality.css
weasyprint dashboard.html -s docs/dashboard-functionality.css -o docs/dashboard-functionality.pdf
```

#### Option 3: Using `marp` (for slide-style PDFs)

```bash
# Install
pip install marp-md

# Convert
marp docs/dashboard-functionality.md -o docs/dashboard-functionality.pdf --css docs/dashboard-functionality.css
```

### Troubleshooting

1. **Mermaid diagrams not rendering**: 
   - Ensure `mermaid-cli` is installed: `npm install -g mermaid-cli`
   - Or use a pandoc filter: `pandoc --filter mermaid-filter`
   - Some PDF generators require pre-rendering mermaid to SVG/PNG

2. **CSS not applying**: 
   - Verify the path to the CSS file is correct
   - Ensure the `--css` flag is placed before `-o` in pandoc

3. **Page breaks causing content splitting**:
   - Add `page-break-before: always;` in the CSS before specific elements
   - Use HTML comments `<!-- pagebreak -->` in the markdown

4. **Fonts missing in PDF**:
   - Use `--pdf-engine=xelatex` or `--pdf-engine=lualatex` which support system fonts
   - Or install the required fonts before conversion

5. **Long tables splitting across pages**:
   - Add `page-break-inside: avoid;` to table styles in the CSS
   - Or break the table into smaller sections

### Output

The final PDF will be generated at:
```
docs/dashboard-functionality.pdf
```

### Recommendations

For best results with this documentation:
1. Use `xelatex` as the PDF engine (supports the Georgia font and special characters)
2. Pre-render mermaid diagrams before PDF conversion if experiencing rendering issues
3. Test the first few pages to adjust margin and font sizes
4. The CSS is designed for A4 paper with 2cm margins