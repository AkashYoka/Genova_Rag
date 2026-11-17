from pathlib import Path

p = Path(r"c:\PC\Yoka\Genova_Rag\frontend\src\components\document\explorer\DocumentExplorer.tsx")
text = p.read_text(encoding="utf-8")
marker = 'from "@/lib/hierarchy-tree-model";'
idx = text.find(marker)
chunk = text[idx:idx+88]
print("CHUNK:", repr(chunk))
print("AROUND 40-55 file:", repr(text[idx+40:idx+60]))

# Just do index insertion without fragile equality
insert_at = text.find('import { openDocChat } from "@/lib/chat-attachments";')
print("openDocChat import at", insert_at)
if insert_at < 0:
    # search nearby
    j = text.find("openDocChat")
    print("openDocChat first", j, repr(text[j-30:j+50]))
