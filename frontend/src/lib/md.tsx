/** Very small markdown renderer (headings, lists, tables, code, emphasis, links).
 *  Deliberately self-contained: no CDN, no external dependency, safe by construction
 *  (it builds React elements, never raw HTML strings). */
import React from "react";

const inline = (text: string, key: string): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  const rx = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|_[^_]+_)/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = rx.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) parts.push(<b key={`${key}-b${i}`}>{tok.slice(2, -2)}</b>);
    else if (tok.startsWith("`")) parts.push(<code key={`${key}-c${i}`}>{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("[")) {
      const mm = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok)!;
      parts.push(<a key={`${key}-a${i}`} href={mm[2]} target={mm[2].startsWith("http") ? "_blank" : undefined} rel="noreferrer">{mm[1]}</a>);
    } else parts.push(<i key={`${key}-i${i}`}>{tok.slice(1, -1)}</i>);
    last = m.index + tok.length; i++;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
};

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let i = 0, key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) { i++; continue; }
    if (/^```/.test(line)) {
      const lang = line.slice(3).trim();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(<pre key={key++} data-lang={lang}>{body.join("\n")}</pre>);
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      out.push(React.createElement(`h${level}`, { key: key++ }, inline(h[2], `h${key}`)));
      i++;
      continue;
    }
    if (/^\|/.test(line)) {
      const rows: string[][] = [];
      while (i < lines.length && /^\|/.test(lines[i])) {
        const cells = lines[i].split("|").slice(1, -1).map((c) => c.trim());
        if (!cells.every((c) => /^-{2,}$/.test(c) || c === "")) rows.push(cells);
        i++;
      }
      const [head, ...body] = rows;
      out.push(
        <div className="tblwrap" key={key++}>
          <table>
            {head && <thead><tr>{head.map((c, j) => <th key={j}>{inline(c, `th${j}`)}</th>)}</tr></thead>}
            <tbody>{body.map((r, j) => <tr key={j}>{r.map((c, k) => <td key={k}>{inline(c, `td${j}-${k}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/, ""));
      const L: any = ordered ? "ol" : "ul";
      out.push(<L key={key++} style={{ marginLeft: 18 }}>{items.map((it, j) => <li key={j}>{inline(it, `li${j}`)}</li>)}</L>);
      continue;
    }
    if (/^>\s?/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) body.push(lines[i++].replace(/^>\s?/, ""));
      out.push(<blockquote key={key++} className="small" style={{ borderLeft: "3px solid #24405f", margin: "8px 0", paddingLeft: 10, color: "#c8d6ee" }}>{inline(body.join(" "), `q${key}`)}</blockquote>);
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) { out.push(<hr key={key++} className="hr" />); i++; continue; }
    const para: string[] = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^[#>|]|^```|^\s*([-*+]|\d+\.)\s/.test(lines[i])) para.push(lines[i++]);
    out.push(<p key={key++}>{inline(para.join(" "), `p${key}`)}</p>);
  }
  return <div className="md">{out}</div>;
}

export async function fetchDocs(): Promise<{ index: any; markdown: Record<string, string> }> {
  const idx = await fetch("/documentation/index.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!idx?.documents?.length) return { index: null, markdown: {} };
  const md: Record<string, string> = {};
  await Promise.all(idx.documents.map(async (d: any) => {
    const t = await fetch(`/documentation/${d.name}.md`).then((r) => (r.ok ? r.text() : "")).catch(() => "");
    md[d.name] = t;
  }));
  return { index: idx, markdown: md };
}
