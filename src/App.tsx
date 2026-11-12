import styles from './App.module.css';

function App() {
  return (
    <main className={styles.shell} aria-labelledby="app-title">
      <section className={styles.panel}>
        <p className={styles.eyebrow}>Todo-1</p>
        <h1 id="app-title" className={styles.title}>
          Todo app foundation
        </h1>
        <p className={styles.copy}>
          A browser-only React and TypeScript app shell is ready for the next
          feature task.
        </p>
      </section>
    </main>
  );
}

export default App;
