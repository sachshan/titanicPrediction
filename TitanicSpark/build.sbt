ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.13"

lazy val root = (project in file("."))
  .settings(
    name := "TitanicSpark"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.5.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.1" % "provided"
)

