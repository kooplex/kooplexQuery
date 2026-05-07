import { useEffect, useMemo, useRef, useState } from 'react'
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml'
import {
  api,
  type ChatHistoryRow,
  type ModelRow,
  type ConfigPayload,
  type QueryResult,
  type ExampleRow,
  type StoredConfig,
  type TableRow,
  type ColumnRow,
  type KnowledgeRow,
  toPayload,
} from './api'
import './App.css'

type View = 'settings' | 'chat' | 'validator' | 'metadata'

type Status = {
  kind: 'success' | 'error' | 'info'
  message: string
} | null

const emptyConfig: ConfigPayload = {
  database_server: {
    hostname: 'localhost',
    port: 5432,
    database: '',
    user: 'postgres',
    password: '',
    type: 'postgresql',
    schema: 'public',
    title: 'Untitled',
    url: '',
  },
  chat_database_server: {
    hostname: 'localhost',
    port: 5432,
    database: '',
    user: 'postgres',
    password: '',
    schema: 'public',
  },
}

type UnknownRecord = Record<string, unknown>

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function firstDefined(record: UnknownRecord, keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) {
      return record[key]
    }
  }
  return undefined
}

function asNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function hasAnyKey(record: UnknownRecord, keys: string[]): boolean {
  return keys.some((key) => key in record)
}

function extractUploadedConfigCandidate(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value[0]
  }

  if (!isRecord(value)) return value

  const wrapped = value.config ?? value.configuration ?? value.payload ?? value.data ?? value.active ?? value.selected_config
  if (wrapped !== undefined) return extractUploadedConfigCandidate(wrapped)

  if (Array.isArray(value.items)) return extractUploadedConfigCandidate(value.items)
  if (Array.isArray(value.configs)) return extractUploadedConfigCandidate(value.configs)

  return value
}

function normalizeUploadedConfig(value: unknown): ConfigPayload | null {
  const candidate = extractUploadedConfigCandidate(value)
  if (!isRecord(candidate)) return null

  const dbRecord = isRecord(candidate.database_server)
    ? candidate.database_server
    : isRecord(candidate.db_cfg)
      ? candidate.db_cfg
      : candidate
  const chatRecord = isRecord(candidate.chat_database_server)
    ? candidate.chat_database_server
    : isRecord(candidate.chat_cfg)
      ? candidate.chat_cfg
      : candidate

  const hasDatabaseShape = hasAnyKey(dbRecord, [
    'hostname',
    'host',
    'db_hostname',
    'url',
    'db_url',
    'connection_string',
    'connectionstring',
    'conn_string',
    'database',
    'db_database',
  ])
  const hasChatShape = hasAnyKey(chatRecord, [
    'hostname',
    'host',
    'chat_hostname',
    'database',
    'chat_database',
    'user',
    'chat_user',
  ])

  if (!hasDatabaseShape && !hasChatShape) {
    return null
  }

  return {
    database_server: {
      hostname: asString(firstDefined(dbRecord, ['hostname', 'host', 'server', 'db_hostname']), emptyConfig.database_server.hostname ?? 'localhost'),
      port: asNumber(firstDefined(dbRecord, ['port', 'db_port']), emptyConfig.database_server.port ?? 5432),
      database: asString(firstDefined(dbRecord, ['database', 'dbname', 'name', 'db_database'])),
      user: asString(firstDefined(dbRecord, ['user', 'username', 'db_user']), emptyConfig.database_server.user ?? 'postgres'),
      password: asString(firstDefined(dbRecord, ['password', 'pass', 'db_password'])),
      type: asString(firstDefined(dbRecord, ['type', 'dialect', 'engine', 'db_type']), emptyConfig.database_server.type ?? 'postgresql'),
      schema: asString(firstDefined(dbRecord, ['schema', 'schema_name', 'db_schema']), emptyConfig.database_server.schema ?? 'public'),
      title: asString(firstDefined(dbRecord, ['title', 'db_title', 'label']), emptyConfig.database_server.title ?? 'Untitled'),
      url: asString(firstDefined(dbRecord, ['url', 'db_url', 'connection_string', 'connectionstring', 'conn_string', 'connectionUrl', 'connection_url'])),
    },
    chat_database_server: {
      hostname: asString(firstDefined(chatRecord, ['hostname', 'host', 'server', 'chat_hostname']), emptyConfig.chat_database_server.hostname ?? 'localhost'),
      port: asNumber(firstDefined(chatRecord, ['port', 'chat_port']), emptyConfig.chat_database_server.port ?? 5432),
      database: asString(firstDefined(chatRecord, ['database', 'dbname', 'name', 'chat_database'])),
      user: asString(firstDefined(chatRecord, ['user', 'username', 'chat_user']), emptyConfig.chat_database_server.user ?? 'postgres'),
      password: asString(firstDefined(chatRecord, ['password', 'pass', 'chat_password'])),
      schema: asString(firstDefined(chatRecord, ['schema', 'schema_name', 'chat_schema']), emptyConfig.chat_database_server.schema ?? 'public'),
    },
  }
}

function parseUploadedConfigText(fileName: string, text: string): unknown {
  const lowerName = fileName.toLowerCase()

  if (lowerName.endsWith('.yaml') || lowerName.endsWith('.yml')) {
    return parseYaml(text)
  }

  return JSON.parse(text)
}

type StreamChunk =
  | { type: 'text'; content: string }
  | { type: 'code'; content: string; language: string }

function parseStreamChunks(content: string): StreamChunk[] {
  if (!content.trim()) return []

  const chunks: StreamChunk[] = []
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g
  let lastIndex = 0

  for (const match of content.matchAll(codeBlockRegex)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      const textChunk = content.slice(lastIndex, index).trim()
      if (textChunk) {
        chunks.push({ type: 'text', content: textChunk })
      }
    }

    const language = (match[1] || '').trim().toLowerCase() || 'text'
    const code = (match[2] || '').trim()
    if (code) {
      chunks.push({ type: 'code', language, content: code })
    }
    lastIndex = index + match[0].length
  }

  if (lastIndex < content.length) {
    const trailing = content.slice(lastIndex).trim()
    if (trailing) {
      chunks.push({ type: 'text', content: trailing })
    }
  }

  return chunks.length > 0 ? chunks : [{ type: 'text', content }]
}

function App() {
  const [view, setView] = useState<View>('settings')
  const [status, setStatus] = useState<Status>(null)
  const [loading, setLoading] = useState(false)

  const [configs, setConfigs] = useState<StoredConfig[]>([])
  const [selectedConfigId, setSelectedConfigId] = useState<number | ''>('')
  const [form, setForm] = useState<ConfigPayload>(emptyConfig)
  const [activeConfig, setActiveConfig] = useState<(ConfigPayload & { id?: number }) | null>(null)

  const [examples, setExamples] = useState<ExampleRow[]>([])
  const [tables, setTables] = useState<TableRow[]>([])
  const [columns, setColumns] = useState<ColumnRow[]>([])
  const [knowledge, setKnowledge] = useState<KnowledgeRow[]>([])
  const [knowledgeReference, setKnowledgeReference] = useState('')
  const [knowledgeContent, setKnowledgeContent] = useState('')
  const [editingKnowledgeId, setEditingKnowledgeId] = useState<number | null>(null)
  const [editingKnowledgeReference, setEditingKnowledgeReference] = useState('')
  const [vectorStats, setVectorStats] = useState<Record<string, number>>({})
  const [vectorQuery, setVectorQuery] = useState('')
  const [vectorCollectionsInput, setVectorCollectionsInput] = useState('')
  const [vectorK, setVectorK] = useState(5)
  const [vectorResults, setVectorResults] = useState<Record<string, Array<{ content: string; metadata: Record<string, unknown>; score: number }>>>({})

  const [sessionId, setSessionId] = useState<number | ''>('')
  const [sessionForm, setSessionForm] = useState({
    username: 'demo-user',
    email: 'demo@example.com',
    label: '',
    meta: '',
  })
  const [history, setHistory] = useState<ChatHistoryRow[]>([])
  const [models, setModels] = useState<ModelRow[]>([])
  const [selectedModel, setSelectedModel] = useState('api')
  const [newModelName, setNewModelName] = useState('')
  const [newModelProvider, setNewModelProvider] = useState('')
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatSql, setChatSql] = useState('select 1 as value')
  const [chatPublic, setChatPublic] = useState(true)
  const [agentResponse, setAgentResponse] = useState('')
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [pendingUploadConnect, setPendingUploadConnect] = useState(false)
  const [uploadedConfigFileName, setUploadedConfigFileName] = useState('')
  const configUploadRef = useRef<HTMLInputElement | null>(null)
  const parsedAgentResponse = useMemo(() => parseStreamChunks(agentResponse), [agentResponse])

  // On mount: load saved configs and restore active config from backend
  useEffect(() => {
    async function init() {
      try {
        const [configsData, activeData, modelsData] = await Promise.all([
          api.listConfigs(),
          api.getActive(),
          api.listModels(),
        ])
        setConfigs(configsData.items)
        setModels(modelsData.items)
        if (activeData.active) {
          setActiveConfig(activeData.active)
          setForm(activeData.active)
          if (activeData.active.id !== undefined) {
            setSelectedConfigId(activeData.active.id)
          }
        }
      } catch {
        // Silently ignore init failures (backend may not be ready)
      }
    }
    init()
  }, [])

  const activeSummary = useMemo(() => {
    if (!activeConfig) return 'No active backend configuration'
    const db = activeConfig.database_server
    return `${db.title ?? 'Untitled'} (${db.hostname}:${db.port}/${db.database})`
  }, [activeConfig])

  async function run<T>(fn: () => Promise<T>, successMsg?: string): Promise<T | undefined> {
    setLoading(true)
    setStatus(null)
    try {
      const data = await fn()
      if (successMsg) setStatus({ kind: 'success', message: successMsg })
      return data
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus({ kind: 'error', message })
      return undefined
    } finally {
      setLoading(false)
    }
  }

  async function refreshConfigs() {
    const data = await run(() => api.listConfigs())
    if (!data) return
    setConfigs(data.items)
  }

  async function refreshActiveConfig() {
    const data = await run(() => api.getActive())
    if (!data) return
    setActiveConfig(data.active)
    if (data.active) setForm(data.active)
  }

  async function loadFromSelected() {
    if (selectedConfigId === '') {
      setStatus({ kind: 'info', message: 'Select a config first.' })
      return
    }

    const selected = configs.find((c) => c.id === selectedConfigId)
    if (!selected) {
      setStatus({ kind: 'error', message: 'Selected config not found.' })
      return
    }

    setForm(toPayload(selected))
    setUploadedConfigFileName('')
    const response = await run(() => api.selectConfig(selected.id), 'Config selected in backend runtime.')
    if (!response) return
    setActiveConfig(response.active)
  }

  async function deleteSelectedConfig() {
    if (selectedConfigId === '') {
      setStatus({ kind: 'info', message: 'Select a config to delete first.' })
      return
    }

    const response = await run(
      () => api.deleteConfig(selectedConfigId),
      'Selected configuration deleted.',
    )
    if (!response) return

    if (activeConfig?.id === selectedConfigId) {
      setActiveConfig(null)
    }
    setSelectedConfigId('')
    await refreshConfigs()
  }

  function downloadConfig(format: 'json' | 'yaml') {
    const data = JSON.stringify(form, null, 2)
    const content = format === 'json' ? data : stringifyYaml(JSON.parse(data))
    const mime = format === 'json' ? 'application/json' : 'application/x-yaml'
    const title = (form.database_server.title ?? 'database_config').trim() || 'database_config'
    const safeTitle = title.replace(/[^A-Za-z0-9_-]+/g, '_')

    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${safeTitle}.${format}`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  async function connectWithPayload(payloadConfig: ConfigPayload, createDb: boolean) {
    const payload = {
      ...payloadConfig,
      save: true,
      create_chat_database_if_missing: createDb,
    }

    const response = await run(
      () => api.connect(payload),
      createDb
        ? 'Connected and chat database ensured.'
        : 'Connected successfully.',
    )

    if (!response) return
    setActiveConfig(response.active)
    await refreshConfigs()
  }

  async function connect(createDb: boolean) {
    await connectWithPayload(form, createDb)
  }

  function triggerConfigUpload(connectAfterUpload: boolean) {
    setPendingUploadConnect(connectAfterUpload)
    configUploadRef.current?.click()
  }

  async function handleConfigUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    const uploadedShouldConnect = pendingUploadConnect
    setPendingUploadConnect(false)

    try {
      const text = await file.text()
      const parsed = parseUploadedConfigText(file.name, text)
      const normalized = normalizeUploadedConfig(parsed)
      if (!normalized) {
        setStatus({ kind: 'error', message: 'Unsupported config format in uploaded file.' })
        return
      }

      setForm(normalized)
      setSelectedConfigId('')
      setView('settings')
      setUploadedConfigFileName(file.name)

      if (uploadedShouldConnect) {
        await connectWithPayload(normalized, false)
      } else {
        const saved = await run(
          () => api.saveConfig(normalized),
          'Config uploaded, saved, and shown in input fields.',
        )
        if (!saved) return
        setSelectedConfigId(saved.id)
        await refreshConfigs()
      }
    } catch {
      setStatus({ kind: 'error', message: 'Failed to read uploaded JSON/YAML config.' })
    }
  }

  async function loadExamples() {
    const data = await run(() => api.listExamples(), 'Validator examples loaded.')
    if (!data) return
    setExamples(data.items)
  }

  async function validateExample(questionId: number) {
    const response = await run(() => api.validateExample(questionId), `Validated question ${questionId}.`)
    if (!response) return
    await loadExamples()
  }

  async function removeExample(questionId: number) {
    const response = await run(() => api.deleteExample(questionId), `Deleted question ${questionId}.`)
    if (!response) return
    await loadExamples()
  }

  async function loadMetadata() {
    const [tablesResponse, columnsResponse, knowledgeResponse] = await Promise.all([
      run(() => api.listTables()),
      run(() => api.listColumns()),
      run(() => api.listKnowledge()),
    ])

    if (tablesResponse) setTables(tablesResponse.items)
    if (columnsResponse) setColumns(columnsResponse.items)
    if (knowledgeResponse) setKnowledge(knowledgeResponse.items)

    if (tablesResponse || columnsResponse || knowledgeResponse) {
      setStatus({ kind: 'success', message: 'Metadata loaded from backend.' })
    }
  }

  async function loadVectorstoreStats() {
    const response = await run(() => api.vectorstoreStats(), 'Vectorstore stats loaded.')
    if (!response) return
    setVectorStats(response.collections)
  }

  async function searchVectorstore() {
    const query = vectorQuery.trim()
    if (!query) {
      setStatus({ kind: 'info', message: 'Vectorstore query is required.' })
      return
    }

    const collections = vectorCollectionsInput
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    const response = await run(
      () => api.vectorstoreSearch({ query, collections: collections.length > 0 ? collections : undefined, k: Math.max(1, vectorK) }),
      'Vectorstore search completed.',
    )
    if (!response) return
    setVectorResults(response.results)
  }

  async function resyncVectorstore() {
    const response = await run(
      () => api.vectorstoreResync(),
      'Vectorstore resynced from examples, metadata, and knowledge.',
    )
    if (!response) return
    await loadVectorstoreStats()
  }

  async function resetVectorstore() {
    const response = await run(
      () => api.vectorstoreReset(),
      'Vectorstore reset completed.',
    )
    if (!response) return
    setVectorResults({})
    await loadVectorstoreStats()
  }

  function clearKnowledgeForm() {
    setKnowledgeReference('')
    setKnowledgeContent('')
    setEditingKnowledgeId(null)
    setEditingKnowledgeReference('')
  }

  function startEditingKnowledge(item: KnowledgeRow) {
    setEditingKnowledgeId(item.id)
    setEditingKnowledgeReference(item.reference)
    setKnowledgeReference(item.reference)
    setKnowledgeContent(item.content ?? '')
  }

  async function saveKnowledge() {
    const reference = knowledgeReference.trim()
    const content = knowledgeContent.trim()

    if (!reference) {
      setStatus({ kind: 'info', message: 'Knowledge reference is required.' })
      return
    }

    if (!content) {
      setStatus({ kind: 'info', message: 'Knowledge content is required.' })
      return
    }

    const response = editingKnowledgeId
      ? await run(
          () => api.updateKnowledge(editingKnowledgeReference || reference, { reference, content }),
          `Knowledge ${reference} updated.`,
        )
      : await run(
          () => api.createKnowledge({ reference, content }),
          `Knowledge ${reference} created.`,
        )

    if (!response) return
    clearKnowledgeForm()
    await loadMetadata()
  }

  async function removeKnowledge(item: KnowledgeRow) {
    const response = await run(
      () => api.deleteKnowledge(item.id),
      `Knowledge ${item.reference} deleted.`,
    )
    if (!response) return

    if (editingKnowledgeId === item.id) {
      clearKnowledgeForm()
    }
    await loadMetadata()
  }

  async function createSession() {
    const response = await run(
      () => api.createSession({ ...sessionForm, referenced_session_id: null }),
      'Chat session created.',
    )
    if (!response) return
    setSessionId(response.session_id)
    const historyResponse = await run(
      () => api.getSessionHistory(response.session_id),
      'Session history loaded.',
    )
    setHistory(historyResponse?.items ?? [])
  }

  async function loadHistory() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set a session id first.' })
      return
    }
    const response = await run(() => api.getSessionHistory(sessionId), 'Session history loaded.')
    if (!response) return
    setHistory(response.items)
  }

  async function loadModels() {
    const response = await run(() => api.listModels(), 'Models loaded.')
    if (!response) return
    setModels(response.items)
    if (response.items.length > 0) {
      setSelectedModel(response.items[0].model_name)
    }
  }

  async function addModel() {
    const modelName = newModelName.trim()
    if (!modelName) {
      setStatus({ kind: 'info', message: 'Model name is required.' })
      return
    }
    const response = await run(
      () => api.addModel({ model_name: modelName, provider: newModelProvider.trim() || null }),
      `Model ${modelName} added.`,
    )
    if (!response) return
    setNewModelName('')
    setNewModelProvider('')
    await loadModels()
  }

  async function deleteSelectedModel() {
    const modelName = selectedModel.trim()
    if (!modelName) {
      setStatus({ kind: 'info', message: 'Select a model first.' })
      return
    }
    const current = models.find((m) => m.model_name === modelName)
    const response = await run(
      () => api.deleteModel(modelName, current?.provider ?? null),
      `Model ${modelName} deleted.`,
    )
    if (!response) return
    await loadModels()
  }

  async function executeQuery() {
    const sql = chatSql.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required.' })
      return
    }
    await executeSqlText(sql)
  }

  async function executeSqlText(sqlText: string) {
    const sql = sqlText.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required.' })
      return
    }

    setChatSql(sql)
    const response = await run(
      () => api.runQuery({ sql }),
      'Query executed successfully.',
    )
    if (!response) return
    setQueryResult(response)
    if (!agentResponse.trim()) {
      setAgentResponse(`Query returned ${response.count} rows.`)
    }
  }

  async function streamChatResponse() {
    const prompt = chatQuestion.trim()
    if (!prompt) {
      setStatus({ kind: 'info', message: 'Question is required for streaming.' })
      return
    }

    setLoading(true)
    setStatus(null)
    setAgentResponse('')

    try {
      const response = await api.streamChat({
        prompt,
        model_name: selectedModel || 'api',
        session_id: sessionId === '' ? null : sessionId,
      })

      if (!response.body) {
        throw new Error('Streaming response body is empty.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (!value) continue
        const chunk = decoder.decode(value, { stream: true })
        if (chunk) {
          setAgentResponse((prev) => `${prev}${chunk}`)
        }
      }

      setStatus({ kind: 'success', message: 'Streaming response completed.' })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus({ kind: 'error', message })
    } finally {
      setLoading(false)
    }
  }

  async function saveQueryRow() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set or create a session first.' })
      return
    }
    const question = chatQuestion.trim()
    const sql = chatSql.trim()
    if (!question || !sql) {
      setStatus({ kind: 'info', message: 'Question and SQL are required to save query.' })
      return
    }
    const response = await run(
      () => api.saveQuery({
        session_id: sessionId,
        question_content: question,
        sql,
        question_type: 'user',
        public: chatPublic,
      }),
      'Query saved.',
    )
    if (!response) return
  }

  async function saveMessagePair() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set or create a session first.' })
      return
    }
    const userPrompt = chatQuestion.trim()
    const responseText = agentResponse.trim()
    if (!userPrompt || !responseText) {
      setStatus({ kind: 'info', message: 'Question and response are required to save messages.' })
      return
    }
    const response = await run(
      () => api.saveMessagePair({
        session_id: sessionId,
        user_prompt: userPrompt,
        agent_response: responseText,
        model_name: selectedModel || 'api',
      }),
      'Message pair saved.',
    )
    if (!response) return
    await loadHistory()
  }

  function updateDb<K extends keyof ConfigPayload['database_server']>(key: K, value: ConfigPayload['database_server'][K]) {
    setForm((prev) => ({
      ...prev,
      database_server: {
        ...prev.database_server,
        [key]: value,
      },
    }))
  }

  function updateChat<K extends keyof ConfigPayload['chat_database_server']>(key: K, value: ConfigPayload['chat_database_server'][K]) {
    setForm((prev) => ({
      ...prev,
      chat_database_server: {
        ...prev.chat_database_server,
        [key]: value,
      },
    }))
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>KooplexQuery Migration UI</h1>
        <div className="active-badge">Active: {activeSummary}</div>
      </header>

      <nav className="nav">
        <button className={view === 'settings' ? 'active' : ''} onClick={() => setView('settings')}>Settings</button>
        <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>Chat</button>
        <button className={view === 'validator' ? 'active' : ''} onClick={() => setView('validator')}>Validator</button>
        <button className={view === 'metadata' ? 'active' : ''} onClick={() => setView('metadata')}>Metadata</button>
      </nav>

      {status && (
        <div className={`status ${status.kind}`}>
          {status.message}
        </div>
      )}

      <main className="panel">
        {view === 'settings' && (
          <section>
            <h2>Backend Settings & Connect</h2>
            <input
              ref={configUploadRef}
              type="file"
              accept="application/json,text/yaml,application/x-yaml,.json,.yaml,.yml"
              className="hidden-input"
              onChange={handleConfigUpload}
            />
            <div className="row-actions">
              <button disabled={loading} onClick={refreshConfigs}>Load saved configs</button>
              <button disabled={loading} onClick={refreshActiveConfig}>Load active config</button>
              <button disabled={loading} onClick={() => triggerConfigUpload(false)}>Upload config file</button>
              <button disabled={loading} onClick={() => triggerConfigUpload(true)}>Upload + connect</button>
            </div>
            {uploadedConfigFileName && (
              <p className="upload-indicator">Loaded from file: {uploadedConfigFileName}</p>
            )}

            <div className="field-row">
              <label>Saved config</label>
              <select
                value={selectedConfigId}
                onChange={(e) => setSelectedConfigId(e.target.value ? Number(e.target.value) : '')}
              >
                <option value="">Select saved configuration</option>
                {configs.map((cfg) => (
                  <option key={cfg.id} value={cfg.id}>
                    {cfg.db_title ?? 'Untitled'} - {cfg.db_hostname}:{cfg.db_port}/{cfg.db_database}
                  </option>
                ))}
              </select>
              <button disabled={loading || selectedConfigId === ''} onClick={loadFromSelected}>Use selected</button>
              <button disabled={loading || selectedConfigId === ''} className="danger" onClick={deleteSelectedConfig}>Delete selected</button>
            </div>

            <div className="row-actions">
              <button disabled={loading} onClick={() => downloadConfig('json')}>Download JSON</button>
              <button disabled={loading} onClick={() => downloadConfig('yaml')}>Download YAML</button>
            </div>

            <div className="grid two-col">
              <fieldset>
                <legend>Source database</legend>
                <label>Title<input value={form.database_server.title ?? ''} onChange={(e) => updateDb('title', e.target.value)} /></label>
                <label>Host<input value={form.database_server.hostname ?? ''} onChange={(e) => updateDb('hostname', e.target.value)} /></label>
                <label>Port<input type="number" value={form.database_server.port ?? 5432} onChange={(e) => updateDb('port', Number(e.target.value) || 5432)} /></label>
                <label>Database<input value={form.database_server.database ?? ''} onChange={(e) => updateDb('database', e.target.value)} /></label>
                <label>User<input value={form.database_server.user ?? ''} onChange={(e) => updateDb('user', e.target.value)} /></label>
                <label>Password<input type="password" value={form.database_server.password ?? ''} onChange={(e) => updateDb('password', e.target.value)} /></label>
                <label>Schema<input value={form.database_server.schema ?? ''} onChange={(e) => updateDb('schema', e.target.value)} /></label>
                <label>Type<input value={form.database_server.type ?? ''} onChange={(e) => updateDb('type', e.target.value)} /></label>
                <label>URL (optional)<input value={form.database_server.url ?? ''} onChange={(e) => updateDb('url', e.target.value)} /></label>
              </fieldset>

              <fieldset>
                <legend>Chat metadata database</legend>
                <label>Host<input value={form.chat_database_server.hostname ?? ''} onChange={(e) => updateChat('hostname', e.target.value)} /></label>
                <label>Port<input type="number" value={form.chat_database_server.port ?? 5432} onChange={(e) => updateChat('port', Number(e.target.value) || 5432)} /></label>
                <label>Database<input value={form.chat_database_server.database ?? ''} onChange={(e) => updateChat('database', e.target.value)} /></label>
                <label>User<input value={form.chat_database_server.user ?? ''} onChange={(e) => updateChat('user', e.target.value)} /></label>
                <label>Password<input type="password" value={form.chat_database_server.password ?? ''} onChange={(e) => updateChat('password', e.target.value)} /></label>
                <label>Schema<input value={form.chat_database_server.schema ?? ''} onChange={(e) => updateChat('schema', e.target.value)} /></label>
              </fieldset>
            </div>

            <div className="row-actions">
              <button disabled={loading} onClick={() => connect(false)}>Connect</button>
              <button disabled={loading} onClick={() => connect(true)}>Create chat DB + connect</button>
            </div>
          </section>
        )}

        {view === 'chat' && (
          <section>
            <h2>Chat</h2>
            <div className="grid two-col">
              <fieldset>
                <legend>Session</legend>
                <label>Session ID
                  <input
                    type="number"
                    value={sessionId}
                    onChange={(e) => setSessionId(e.target.value ? Number(e.target.value) : '')}
                  />
                </label>
                <label>Username
                  <input
                    value={sessionForm.username}
                    onChange={(e) => setSessionForm((prev) => ({ ...prev, username: e.target.value }))}
                  />
                </label>
                <label>Email
                  <input
                    value={sessionForm.email}
                    onChange={(e) => setSessionForm((prev) => ({ ...prev, email: e.target.value }))}
                  />
                </label>
                <label>Label
                  <input
                    value={sessionForm.label}
                    onChange={(e) => setSessionForm((prev) => ({ ...prev, label: e.target.value }))}
                  />
                </label>
                <label>Meta
                  <input
                    value={sessionForm.meta}
                    onChange={(e) => setSessionForm((prev) => ({ ...prev, meta: e.target.value }))}
                  />
                </label>
                <div className="row-actions">
                  <button disabled={loading} onClick={createSession}>Create session</button>
                  <button disabled={loading} onClick={loadHistory}>Load history</button>
                </div>
              </fieldset>

              <fieldset>
                <legend>Models</legend>
                <label>Selected model
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="">Select model</option>
                    {models.map((m, idx) => (
                      <option key={`${m.model_name}-${m.provider ?? 'none'}-${idx}`} value={m.model_name}>
                        {m.provider ? `${m.provider}/` : ''}{m.model_name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>New model name
                  <input
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                  />
                </label>
                <label>New model provider
                  <input
                    value={newModelProvider}
                    onChange={(e) => setNewModelProvider(e.target.value)}
                  />
                </label>
                <div className="row-actions">
                  <button disabled={loading} onClick={loadModels}>Load models</button>
                  <button disabled={loading} onClick={addModel}>Add model</button>
                  <button disabled={loading} className="danger" onClick={deleteSelectedModel}>Delete selected</button>
                </div>
              </fieldset>
            </div>

            <fieldset>
              <legend>Query + message workflow</legend>
              <label>Question
                <input
                  value={chatQuestion}
                  onChange={(e) => setChatQuestion(e.target.value)}
                  placeholder="Natural language question"
                />
              </label>
              <label>SQL
                <textarea
                  className="code-area"
                  value={chatSql}
                  onChange={(e) => setChatSql(e.target.value)}
                />
              </label>
              <div className="grid two-col">
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={chatPublic}
                    onChange={(e) => setChatPublic(e.target.checked)}
                  />
                  Save query as public
                </label>
              </div>
              <label>Agent response
                <textarea
                  className="code-area"
                  value={agentResponse}
                  onChange={(e) => setAgentResponse(e.target.value)}
                  placeholder="Assistant answer summary"
                />
              </label>
              {parsedAgentResponse.length > 0 && (
                <div className="stream-renderer">
                  {parsedAgentResponse.map((chunk, idx) => {
                    if (chunk.type === 'text') {
                      return (
                        <article className="stream-chunk stream-chunk-text" key={`stream-text-${idx}`}>
                          <pre>{chunk.content}</pre>
                        </article>
                      )
                    }

                    const isSql = chunk.language === 'sql'
                    return (
                      <article className="stream-chunk stream-chunk-code" key={`stream-code-${idx}`}>
                        <div className="stream-chunk-header">
                          <strong>{isSql ? 'SQL block' : `Code block (${chunk.language})`}</strong>
                        </div>
                        <pre>{chunk.content}</pre>
                        {isSql && (
                          <div className="row-actions">
                            <button
                              disabled={loading}
                              onClick={() => setChatSql(chunk.content)}
                            >
                              Use SQL in editor
                            </button>
                            <button
                              disabled={loading}
                              onClick={() => executeSqlText(chunk.content)}
                            >
                              Run this SQL
                            </button>
                          </div>
                        )}
                      </article>
                    )
                  })}
                </div>
              )}
              <div className="row-actions">
                <button disabled={loading} onClick={streamChatResponse}>Stream response</button>
                <button disabled={loading} onClick={executeQuery}>Run SQL</button>
                <button disabled={loading} onClick={saveQueryRow}>Save query</button>
                <button disabled={loading} onClick={saveMessagePair}>Save message pair</button>
              </div>
            </fieldset>

            {queryResult && (
              <article>
                <h3>Query result ({queryResult.count})</h3>
                <div className="result-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        {queryResult.columns.map((columnName, idx) => (
                          <th key={`col-${idx}`}>{columnName}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {queryResult.rows.map((row, idx) => (
                        <tr key={`row-${idx}`}>
                          <td>{idx + 1}</td>
                          {row.map((cell, cellIdx) => (
                            <td key={`cell-${idx}-${cellIdx}`}>
                              <pre>{cell === null || cell === undefined ? '' : String(cell)}</pre>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            )}

            <article>
              <h3>Chat history ({history.length})</h3>
              <ul className="list history-list">
                {history.map((item) => {
                  const isSystem = item.role === 'system'
                  return (
                    <li key={item.id} className={isSystem ? 'history-item-system' : ''}>
                      <strong>{isSystem ? 'Initial context/instruction' : `${item.sequence}. ${item.role}`}</strong>
                      <span>{item.timestamp || (isSystem ? 'Preloaded' : '-')}</span>
                      <pre>{item.content}</pre>
                    </li>
                  )
                })}
              </ul>
            </article>
          </section>
        )}

        {view === 'validator' && (
          <section>
            <h2>Validator</h2>
            <div className="row-actions">
              <button disabled={loading} onClick={loadExamples}>Load examples</button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Question</th>
                  <th>Score</th>
                  <th>Public</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {examples.map((row) => (
                  <tr key={row.question_id}>
                    <td>{row.question_id}</td>
                    <td>{row.type}</td>
                    <td title={row.question_content}>{row.question_content}</td>
                    <td>{row.score ?? '-'}</td>
                    <td>{String(row.public)}</td>
                    <td className="cell-actions">
                      <button disabled={loading} onClick={() => validateExample(row.question_id)}>Validate</button>
                      <button disabled={loading} className="danger" onClick={() => removeExample(row.question_id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {view === 'metadata' && (
          <section>
            <h2>Metadata</h2>
            <div className="row-actions">
              <button disabled={loading} onClick={loadMetadata}>Load metadata</button>
              <button disabled={loading} onClick={loadVectorstoreStats}>Load vectorstore stats</button>
            </div>

            <fieldset>
              <legend>Vectorstore</legend>
              <label>Search query
                <input
                  value={vectorQuery}
                  onChange={(e) => setVectorQuery(e.target.value)}
                  placeholder="Find similar examples or schema hints"
                />
              </label>
              <div className="grid two-col">
                <label>Collections (comma-separated, optional)
                  <input
                    value={vectorCollectionsInput}
                    onChange={(e) => setVectorCollectionsInput(e.target.value)}
                    placeholder="examples, docs"
                  />
                </label>
                <label>Top K
                  <input
                    type="number"
                    value={vectorK}
                    onChange={(e) => setVectorK(Number(e.target.value) || 1)}
                  />
                </label>
              </div>
              <div className="row-actions">
                <button disabled={loading} onClick={searchVectorstore}>Search vectorstore</button>
                <button disabled={loading} onClick={resyncVectorstore}>Resync vectorstore</button>
                <button disabled={loading} className="danger" onClick={resetVectorstore}>Reset vectorstore</button>
              </div>
              <pre>{JSON.stringify(vectorStats, null, 2)}</pre>
            </fieldset>

            <fieldset>
              <legend>Knowledge Manager</legend>
              <label>Reference
                <input
                  value={knowledgeReference}
                  onChange={(e) => setKnowledgeReference(e.target.value)}
                  placeholder="schema | instruction | business_rules"
                />
              </label>
              <label>Content
                <textarea
                  className="code-area"
                  value={knowledgeContent}
                  onChange={(e) => setKnowledgeContent(e.target.value)}
                  placeholder="Paste schema, instructions, or notes"
                />
              </label>
              <div className="row-actions">
                <button disabled={loading} onClick={saveKnowledge}>
                  {editingKnowledgeId ? 'Update knowledge' : 'Add knowledge'}
                </button>
                <button disabled={loading} onClick={clearKnowledgeForm}>Clear</button>
              </div>
            </fieldset>

            <div className="grid three-col">
              <article>
                <h3>Tables ({tables.length})</h3>
                <ul className="list">
                  {tables.map((t, idx) => (
                    <li key={`${t.table}-${idx}`}>
                      <strong>{t.table}</strong>
                      <span>{t.description ?? '(no description)'}</span>
                    </li>
                  ))}
                </ul>
              </article>

              <article>
                <h3>Columns ({columns.length})</h3>
                <ul className="list">
                  {columns.map((c, idx) => (
                    <li key={`${c.table}-${c.column}-${idx}`}>
                      <strong>{c.table}.{c.column}</strong>
                      <span>{c.type}</span>
                    </li>
                  ))}
                </ul>
              </article>

              <article>
                <h3>Knowledge ({knowledge.length})</h3>
                <ul className="list">
                  {knowledge.map((k) => (
                    <li key={k.id}>
                      <strong>{k.reference}</strong>
                      <span>{k.content ?? '(empty)'}</span>
                      <div className="row-actions">
                        <button disabled={loading} onClick={() => startEditingKnowledge(k)}>Edit</button>
                        <button disabled={loading} className="danger" onClick={() => removeKnowledge(k)}>Delete</button>
                      </div>
                    </li>
                  ))}
                </ul>
              </article>
            </div>

            <article>
              <h3>Vectorstore Search Results</h3>
              <pre>{JSON.stringify(vectorResults, null, 2)}</pre>
            </article>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
