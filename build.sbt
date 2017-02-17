name := "MPML"

version := "1.0"

scalaVersion := "2.11.8"

// add spark dependencies
val sparkVersion = "2.1.0"
libraryDependencies ++= Seq(
  //  groupID % artifactID % revision,
  //  %% will add scala version to artifactID
  //  if necessary, using % "provided" to exclude spark from jars
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.log4s" %% "log4s" % "1.3.3",
  "junit" % "junit" % "4.12",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

assemblyJarName in assembly := "mpml.jar"

mainClass in assembly := Some("classification.MyLogisticRegression")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
    