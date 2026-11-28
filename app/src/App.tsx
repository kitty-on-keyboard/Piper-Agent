import React, { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApprovalRequestNotification, SessionSummary } from "./protocol.generated";
import { bridge, WorkspaceDiffResult } from "./lib/bridge";
import { createInitialSessionState, sessionReducer } from "./lib/reducer";
import {
  AppSettings,
  DEFAULT_SETTINGS,
  mergeLoadedSettings,
  pushRecentWorkspace,
  resolveDraftModelDir,
  ScannedModel,
  toRunSettings,
} from "./lib/settings";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { WelcomeView } from "./components/WelcomeView";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { InspectorDrawer } from "./components/InspectorDrawer";
import { SettingsModal } from "./components/SettingsModal";

export const App: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [scannedModels, setScannedModels] = useState<ScannedModel[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [diff, setDiff] = useState<WorkspaceDiffResult>({ is_git: false, branch: "", files: [] });
  const [workspaceFiles, setWorkspaceFiles] = useState<string[]>([]);
  const [sidecarRunning, setSidecarRunning] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [composerText, setComposerText] = useState("");
  const [attachedImages, setAttachedImages] = useState<string[]>([]);

  const [session, dispatch] = useReducer(sessionReducer, undefined, createInitialSessionState);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const scannedRef = useRef(scannedModels);
  scannedRef.current = scannedModels;

  // Apply light/dark class on <html>
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const sync = () => {
      const wantDark =
        settings.theme === "dark" || (settings.theme === "system" && mq.matches);
      document.documentElement.classList.toggle("dark", wantDark);
    };
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [settings.theme]);

  const persistSettings = useCallback((next: AppSettings) => {
    setSettings(next);
    void bridge.saveSettings(next);
  }, []);

  const refreshWorkspaceData = useCallback(async (wsRoot: string) => {
    if (!wsRoot) {
      setSessions([]);
      setDiff({ is_git: false, branch: "", files: [] });
      setWorkspaceFiles([]);
      return;
    }
    const [diffRes, filesRes, sessRes] = await Promise.allSettled([
      bridge.workspaceDiff(wsRoot),
      bridge.workspaceFiles(wsRoot, 2000),
      bridge.listSessions({ workspace_root: wsRoot, limit: 40 }),
    ]);
    if (diffRes.status === "fulfilled") setDiff(diffRes.value);
    if (filesRes.status === "fulfilled") setWorkspaceFiles(filesRes.value.files);
    if (sessRes.status === "fulfilled") setSessions(sessRes.value.sessions);
  }, []);

  const rescanModels = useCallback(async (roots: string[]) => {
    try {
      const res = await bridge.scanModels(roots);
      setScannedModels(res.models ?? []);
      return res.models ?? [];
    } catch {
      return [];
    }
  }, []);

  const handlePickWorkspace = useCallback(async () => {
    const res = await bridge.pickFolder("Open Workspace Repository", settingsRef.current.workspace_root);
    if (!res.cancelled && res.path) {
      const next: AppSettings = {
        ...settingsRef.current,
        workspace_root: res.path,
        recent_workspaces: pushRecentWorkspace(settingsRef.current.recent_workspaces, res.path),
      };
      persistSettings(next);
      dispatch({ type: "reset" });
      void refreshWorkspaceData(res.path);
    }
  }, [persistSettings, refreshWorkspaceData]);

  // Subscribe to native + sidecar events and run initial host handshake.
  useEffect(() => {
    const unsub = bridge.subscribe((ev) => {
      switch (ev.method) {
        case "lmp/ready":
          setSidecarRunning(true);
          break;
        case "lmp/model_status":
          dispatch({ type: "model_status", note: ev.params });
          break;
        case "lmp/token":
          dispatch({ type: "token", note: ev.params, nowMs: Date.now() });
          break;
        case "lmp/turn":
          dispatch({ type: "turn", note: ev.params, nowMs: Date.now() });
          if (ev.params.edit_bytes > 0 && settingsRef.current.workspace_root) {
            void bridge
              .workspaceDiff(settingsRef.current.workspace_root)
              .then(setDiff)
              .catch(() => {});
          }
          break;
        case "lmp/checklist":
          dispatch({ type: "checklist", note: ev.params });
          break;
        case "lmp/plan_ready":
          dispatch({ type: "plan_ready", note: ev.params });
          break;
        case "lmp/verification":
          dispatch({ type: "verification", note: ev.params });
          break;
        case "lmp/approval_request":
          dispatch({ type: "approval_request", note: ev.params });
          break;
        case "lmp/perf":
          dispatch({ type: "perf", note: ev.params });
          break;
        case "lmp/notice":
          dispatch({ type: "notice", note: ev.params });
          break;
        case "lmp/run_end":
          dispatch({ type: "run_end", note: ev.params, nowMs: Date.now() });
          if (settingsRef.current.workspace_root) {
            void refreshWorkspaceData(settingsRef.current.workspace_root);
          }
          break;
        case "host/sidecar_exit":
          setSidecarRunning(false);
          dispatch({
            type: "local_error",
            text: `Sidecar exited (code ${ev.params.code}). ${
              ev.params.stderr_tail.slice(-3).join(" | ") || "Use Engine -> Restart sidecar."
            }`,
          });
          break;
        case "host/shortcut":
          if (ev.params.name === "settings") setSettingsOpen(true);
          else if (ev.params.name === "toggle_sidebar") setSidebarOpen((v) => !v);
          else if (ev.params.name === "toggle_inspector") setInspectorOpen((v) => !v);
          else if (ev.params.name === "new_session") dispatch({ type: "reset" });
          else if (ev.params.name === "open_workspace") void handlePickWorkspace();
          break;
      }
    });

    void (async () => {
      try {
        const init = await bridge.hostInit();
        setSidecarRunning(init.sidecar_running);
        const merged = mergeLoadedSettings(init.settings, init.qwen_dir);
        persistSettings(merged);
        const found = await rescanModels([
          ...merged.model_scan_roots,
          ...(merged.model_dir ? [merged.model_dir] : []),
        ]);
        if (!merged.model_dir && found.length > 0) {
          persistSettings({
            ...merged,
            model_dir: found[0].path,
            draft_model_dir: found[0].kind === "dense" ? found[0].mtp_path : "",
          });
        }
        if (merged.workspace_root) {
          void refreshWorkspaceData(merged.workspace_root);
        }
      } catch (err) {
        dispatch({
          type: "local_error",
          text: err instanceof Error ? err.message : "Host initialization failed",
        });
      }
    })();

    return unsub;
  }, [handlePickWorkspace, persistSettings, refreshWorkspaceData, rescanModels]);

  const handleLoadModel = useCallback(async (modelDir: string, draftDir = "") => {
    if (!modelDir) {
      setSettingsOpen(true);
      return;
    }
    try {
      const res = await bridge.loadModel({ model_dir: modelDir, draft_model_dir: draftDir });
      if (!res.loaded && res.error) {
        dispatch({ type: "local_error", text: res.error });
      }
    } catch (err) {
      dispatch({
        type: "local_error",
        text: err instanceof Error ? err.message : "Failed to load model",
      });
    }
  }, []);

  const handleSubmitText = useCallback(
    async (explicitText?: string, overrideMode?: AppSettings["mode"]) => {
      const prompt = (explicitText ?? composerText).trim();
      if (!prompt) return;

      const cur = settingsRef.current;
      const effectiveSettings = overrideMode ? { ...cur, mode: overrideMode } : cur;
      if (overrideMode && overrideMode !== cur.mode) {
        persistSettings(effectiveSettings);
      }

      if (!effectiveSettings.workspace_root) {
        await handlePickWorkspace();
        return;
      }
      if (!effectiveSettings.model_dir) {
        setSettingsOpen(true);
        return;
      }

      const imgs = explicitText ? [] : attachedImages;
      if (!explicitText) {
        setComposerText("");
        setAttachedImages([]);
      }

      const runSettings = toRunSettings(effectiveSettings, scannedRef.current);

      try {
        if (!session.activeRunId) {
          const started = await bridge.startRun({
            mission: prompt,
            settings: runSettings,
            image_paths: imgs,
          });
          dispatch({
            type: "user_start",
            runId: started.run_id,
            text: prompt,
            images: imgs,
          });
        } else {
          const msgRes = await bridge.sendMessage({
            run_id: session.activeRunId,
            text: prompt,
            image_paths: imgs,
            settings: runSettings,
          });
          dispatch({
            type: "user_followup",
            runId: msgRes.run_id || session.activeRunId,
            text: prompt,
            images: imgs,
            steering: session.running,
          });
        }
      } catch (err) {
        dispatch({
          type: "local_error",
          text: err instanceof Error ? err.message : "Failed sending prompt",
        });
      }
    },
    [attachedImages, composerText, handlePickWorkspace, persistSettings, session.activeRunId, session.running]
  );

  const handleDecideApproval = useCallback(
    async (opts: {
      request: ApprovalRequestNotification;
      approved: boolean;
      allowWritesForRun: boolean;
      allowCommandForRun: boolean;
      persistCommand: boolean;
    }) => {
      try {
        await bridge.approveTool({
          request_id: opts.request.request_id,
          approved: opts.approved,
          allow_writes_for_run: opts.allowWritesForRun,
          allow_command_for_run: opts.allowCommandForRun,
        });
        dispatch({
          type: "approval_resolved",
          requestId: opts.request.request_id,
          resolution: opts.approved ? "approved" : "denied",
        });
        if (opts.persistCommand && opts.request.command) {
          const cur = settingsRef.current;
          if (!cur.allowed_commands.includes(opts.request.command)) {
            persistSettings({
              ...cur,
              allowed_commands: [...cur.allowed_commands, opts.request.command],
            });
          }
        }
      } catch (err) {
        dispatch({
          type: "local_error",
          text: err instanceof Error ? err.message : "Approval failed",
        });
      }
    },
    [persistSettings]
  );

  const handleResumeSession = useCallback(async (runId: string) => {
    try {
      const res = await bridge.resumeSession({
        run_id: runId,
        settings: toRunSettings(settingsRef.current, scannedRef.current),
      });
      if (!res.ok) {
        dispatch({ type: "local_error", text: res.why || "Unable to resume session" });
        return;
      }
      dispatch({ type: "resumed", res });
    } catch (err) {
      dispatch({
        type: "local_error",
        text: err instanceof Error ? err.message : "Resume failed",
      });
    }
  }, []);

  const title =
    sessions.find((s) => s.run_id === session.activeRunId)?.mission ||
    (settings.workspace_root ? settings.workspace_root.split("/").pop() || "Piper" : "Piper");

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-canvas)] text-[var(--ink)]">
      <Sidebar
        open={sidebarOpen}
        settings={settings}
        branch={diff.branch}
        sessions={sessions}
        activeRunId={session.activeRunId}
        modelStatus={session.modelStatus}
        scannedModels={scannedModels}
        onSelectWorkspace={(ws) => {
          persistSettings({
            ...settings,
            workspace_root: ws,
            recent_workspaces: pushRecentWorkspace(settings.recent_workspaces, ws),
          });
          dispatch({ type: "reset" });
          void refreshWorkspaceData(ws);
        }}
        onPickWorkspace={() => void handlePickWorkspace()}
        onNewSession={() => dispatch({ type: "reset" })}
        onResumeSession={(id) => void handleResumeSession(id)}
        onSelectModel={(modelPath, loadImmediately) => {
          const found = scannedModels.find((m) => m.path === modelPath);
          const draft = found?.kind === "dense" ? found.mtp_path : "";
          const next = { ...settings, model_dir: modelPath, draft_model_dir: draft };
          persistSettings(next);
          if (loadImmediately) {
            void handleLoadModel(modelPath, resolveDraftModelDir(next, scannedModels));
          }
        }}
        onUnloadModel={() => void bridge.unloadModel()}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          sidebarOpen={sidebarOpen}
          inspectorOpen={inspectorOpen}
          title={title}
          running={session.running}
          perf={session.perf}
          changedFilesCount={diff.files.length}
          sandboxTier={settings.mode === "plan" ? 0 : settings.sandbox_tier}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
          onToggleInspector={() => setInspectorOpen((v) => !v)}
        />

        <main className="min-h-0 flex-1 overflow-y-auto">
          {session.entries.length === 0 ? (
            <WelcomeView
              workspaceRoot={settings.workspace_root}
              branch={diff.branch}
              modelDir={settings.model_dir}
              modelStatus={session.modelStatus}
              scannedModels={scannedModels}
              onPickWorkspace={() => void handlePickWorkspace()}
              onLoadModel={() =>
                void handleLoadModel(
                  settings.model_dir,
                  resolveDraftModelDir(settings, scannedModels)
                )
              }
              onOpenSettings={() => setSettingsOpen(true)}
              onUsePrompt={(prompt) => setComposerText(prompt)}
            />
          ) : (
            <Transcript
              entries={session.entries}
              running={session.running}
              onDecideApproval={handleDecideApproval}
              onExecutePlan={(planText) =>
                void handleSubmitText(
                  `Execute the approved plan:\n\n${planText}`,
                  "agent"
                )
              }
            />
          )}
        </main>

        <Composer
          text={composerText}
          onTextChange={setComposerText}
          mode={settings.mode}
          onModeChange={(mode) => persistSettings({ ...settings, mode })}
          reasoningEffort={settings.reasoning_effort}
          onReasoningEffortChange={(reasoning_effort) =>
            persistSettings({ ...settings, reasoning_effort })
          }
          supportsReasoningEffort={session.modelStatus.supports_reasoning_effort}
          running={session.running}
          workspaceRoot={settings.workspace_root}
          workspaceFiles={workspaceFiles}
          attachedImages={attachedImages}
          onAttachedImagesChange={setAttachedImages}
          onSubmit={() => void handleSubmitText()}
          onCancel={() => {
            if (session.activeRunId) {
              void bridge.cancelRun({ run_id: session.activeRunId });
            }
          }}
        />
      </div>

      <InspectorDrawer
        open={inspectorOpen}
        workspaceRoot={settings.workspace_root}
        diff={diff}
        files={workspaceFiles}
        perf={session.perf}
        sandboxTier={settings.mode === "plan" ? 0 : settings.sandbox_tier}
        verifyContract={settings.verify_contract}
        sidecarRunning={sidecarRunning}
        onRefreshDiff={() => void refreshWorkspaceData(settings.workspace_root)}
        onRestartSidecar={() => {
          void bridge.restartSidecar().then((r) => {
            setSidecarRunning(r.ok);
            if (!r.ok && r.error) dispatch({ type: "local_error", text: r.error });
          });
        }}
      />

      <SettingsModal
        open={settingsOpen}
        settings={settings}
        scannedModels={scannedModels}
        onClose={() => setSettingsOpen(false)}
        onUpdate={persistSettings}
        onRescanModels={() =>
          void rescanModels([
            ...settings.model_scan_roots,
            ...(settings.model_dir ? [settings.model_dir] : []),
          ])
        }
        onLoadModelNow={(mDir, dDir) => {
          setSettingsOpen(false);
          void handleLoadModel(mDir, dDir);
        }}
      />
    </div>
  );
};
