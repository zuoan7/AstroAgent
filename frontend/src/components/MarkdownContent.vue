<template>
  <div class="markdown-content" v-html="renderedHtml"></div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  content: { type: String, default: '' },
})

const renderedHtml = computed(() => {
  const source = String(props.content || '').trim()
  if (!source) {
    return ''
  }
  return renderMarkdown(source)
})

function renderMarkdown(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n')
  const blocks = []
  let paragraph = []

  function flushParagraph() {
    if (!paragraph.length) {
      return
    }
    blocks.push(`<p>${renderInline(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`)
    paragraph = []
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const trimmed = line.trim()

    if (!trimmed) {
      flushParagraph()
      continue
    }

    const fence = trimmed.match(/^```([a-zA-Z0-9_-]+)?\s*$/)
    if (fence) {
      flushParagraph()
      const codeLines = []
      index += 1
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index])
        index += 1
      }
      const language = fence[1] ? ` data-language="${escapeHtml(fence[1])}"` : ''
      blocks.push(`<pre${language}><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`)
      continue
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/)
    if (heading) {
      flushParagraph()
      const level = heading[1].length
      blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`)
      continue
    }

    const quote = trimmed.match(/^>\s?(.*)$/)
    if (quote) {
      flushParagraph()
      const quoteLines = [quote[1]]
      while (index + 1 < lines.length) {
        const next = lines[index + 1].trim().match(/^>\s?(.*)$/)
        if (!next) {
          break
        }
        quoteLines.push(next[1])
        index += 1
      }
      blocks.push(`<blockquote>${renderMarkdown(quoteLines.join('\n'))}</blockquote>`)
      continue
    }

    const unordered = line.match(/^\s*[-*+]\s+(.+)$/)
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/)
    if (unordered || ordered) {
      flushParagraph()
      const tag = unordered ? 'ul' : 'ol'
      const items = []
      let currentMatch = unordered || ordered
      while (currentMatch) {
        items.push(`<li>${renderInline(currentMatch[1])}</li>`)
        if (index + 1 >= lines.length) {
          break
        }
        const nextLine = lines[index + 1]
        const nextMatch = tag === 'ul'
          ? nextLine.match(/^\s*[-*+]\s+(.+)$/)
          : nextLine.match(/^\s*\d+[.)]\s+(.+)$/)
        if (!nextMatch) {
          break
        }
        currentMatch = nextMatch
        index += 1
      }
      blocks.push(`<${tag}>${items.join('')}</${tag}>`)
      continue
    }

    paragraph.push(line)
  }

  flushParagraph()
  return blocks.join('')
}

function renderInline(source) {
  const codeSpans = []
  let text = source.replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE_${codeSpans.length}@@`
    codeSpans.push(`<code>${escapeHtml(code)}</code>`)
    return token
  })

  text = escapeHtml(text)
  text = text.replace(
    /\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g,
    (_, label, url) => `<a href="${escapeAttribute(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
  )
  text = text
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')

  codeSpans.forEach((html, index) => {
    text = text.replace(`@@CODE_${index}@@`, html)
  })
  return text
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, '&#96;')
}
</script>
